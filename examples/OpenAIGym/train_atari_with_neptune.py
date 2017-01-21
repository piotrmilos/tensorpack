#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import json
import multiprocessing
import uuid

import numpy as np
import time
from deepsense import neptune
from six.moves import queue

import examples.OpenAIGym.common
from examples.OpenAIGym.common import (play_model, Evaluator, eval_model_multithread)
from tensorpack import *
from tensorpack.RL import *
from tensorpack.RL.common import MapPlayerState, PreventStuckPlayer, LimitLengthPlayer
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.RL.simulator import SimulatorMaster, SimulatorProcess, TransitionExperience
from tensorpack.callbacks.base import Callback, PeriodicCallback
from tensorpack.callbacks.common import ModelSaver
from tensorpack.callbacks.concurrency import StartProcOrThread
from tensorpack.callbacks.group import Callbacks
from tensorpack.callbacks.neptune import NeptuneLogger
from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter, NeputneHyperParamSetter
from tensorpack.callbacks.stats import StatPrinter
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.raw import DataFromQueue
from tensorpack.models.fc import FullyConnected
from tensorpack.models.model_desc import ModelDesc, InputVar, get_current_tower_context
from tensorpack.models.nonlin import PReLU
from tensorpack.predict.common import PredictConfig
from tensorpack.predict.concurrency import MultiThreadAsyncPredictor
from tensorpack.tfutils import summary
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.tfutils.gradproc import SummaryGradient, MapGradient
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.config import TrainConfig
from tensorpack.train.multigpu import AsyncMultiGPUTrainer
from tensorpack.utils.concurrency import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.serialize import *
import examples.OpenAIGym.non_makov_test as nmt

# TODO: PM This is the version for the simplistic pong
IMAGE_SIZE = (3, 2)
# IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99

CHANNEL = FRAME_HISTORY * 1
# CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5

# STEP_PER_EPOCH = 2000
# EVAL_EPISODE = 20
STEP_PER_EPOCH = 6000
EVAL_EPISODE = 5

BATCH_SIZE = 128
SIMULATOR_PROC = 30
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None
EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = None
ENV_NAME = None
EXPERIMENT_MODEL = None

EXPERIMENT_ID = None

DEBUGING_INFO = 1


def get_player(viz=False, train=False, dumpdir=None):
    # pl = None
    # if ENV_NAME == "MultiRoomEnv":
    #     print "An awful hack for the moment"
    #     pl = nmt.NonMarkovEnvironment()
    # else:
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    # print "HAS ATTR:{}".format(hasattr(pl, "experiment_id"))
    if EXPERIMENT_ID != None and hasattr(pl.gymenv, "set_experiment_id"):
        pl.gymenv.set_experiment_id(EXPERIMENT_ID)
    
    pl = MapPlayerState(pl, EXPERIMENT_MODEL.get_screen_processor())

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()


    if hasattr(EXPERIMENT_MODEL, "get_history_processor"):
        pl = HistoryFramePlayer(pl, FRAME_HISTORY, EXPERIMENT_MODEL.get_history_processor())
    else:
        pl = HistoryFramePlayer(pl, FRAME_HISTORY, )
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl
examples.OpenAIGym.common.get_player = get_player

class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE*8*2)

    def _setup_graph(self):
        self.sess = self.trainer.sess
        self.async_predictor = MultiThreadAsyncPredictor(
                self.trainer.get_predict_funcs(['state'], ['logitsT', 'pred_value'],
                PREDICTOR_THREAD), batch_size=15)
        self.async_predictor.run()

    def _on_state(self, state, ident):
        def cb(outputs):
            distrib, value = outputs.result()
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client = self.clients[ident]
            client.memory.append(TransitionExperience(state, action, None, value=value))
            self.send_queue.put([ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False)

    def _parse_memory(self, init_r, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            # print "Clipping: {}".format(R)
            self.queue.put([k.state, k.action, R])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

def get_atribute(ctx, name, default_value = None):
    # type: (Neptune.Context, string, object) -> object

    if hasattr(ctx.params, name):
        return getattr(ctx.params, name)
    else:
        return default_value


def set_motezuma_env_options(str, file):
    with open(file, "w") as file:
        file.write(str)


def get_config(ctx):
    """ We use addiional id to make it possible to run multiple instances of the same code
    We use the neputne id for an easy reference.
    piotr.milos@codilime
    """
    global HISTORY_LOGS, EXPERIMENT_ID #Ugly hack, make it better at some point, may be ;)
    id = ctx.job.id
    EXPERIMENT_ID = hash(id)

    import montezuma_env

    ctx.job.register_action("Set starting point procssor:",
                            lambda str: set_motezuma_env_options(str, montezuma_env.STARTING_POINT_SELECTOR))
    ctx.job.register_action("Set rewards:",
                            lambda str: set_motezuma_env_options(str, montezuma_env.REWARDS_FILE))

    logger.auto_set_dir(suffix=id)

    # (self, parameters, number_of_actions, input_shape)

    M = EXPERIMENT_MODEL

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR_{}'.format(id), '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}-{}'.format(PIPE_DIR, name_base, id)
    names2c = 'ipc://{}/sim-s2c-{}-{}'.format(PIPE_DIR, name_base, id)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)

    # My stuff - PM
    neptuneLogger = NeptuneLogger.get_instance()
    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)
    num_epochs = get_atribute(ctx, "num_epochs", 100)

    rewards_str = get_atribute(ctx, "rewards", "5 1 -200")
    with open(montezuma_env.REWARDS_FILE, "w") as file:
        file.write(rewards_str)


    if hasattr(ctx.params, "learning_rate_schedule"):
        schedule_str = str(ctx.params.learning_rate_schedule)
    else: #Default value inhereted from tensorpack
        schedule_str = "[[80, 0.0003], [120, 0.0001]]"
    logger.info("Setting learing rate schedule:{}".format(schedule_str))
    learning_rate_scheduler = ScheduledHyperParamSetter('learning_rate', json.loads(schedule_str))

    if hasattr(ctx.params, "entropy_beta_schedule"):
        schedule_str = str(ctx.params.entropy_beta_schedule)
    else: #Default value inhereted from tensorpack
        schedule_str = "[[80, 0.0003], [120, 0.0001]]"
    logger.info("Setting entropy beta schedule:{}".format(schedule_str))
    entropy_beta_scheduler = ScheduledHyperParamSetter('entropy_beta', json.loads(schedule_str))

    if hasattr(ctx.params, "explore_factor_schedule"):
        schedule_str = str(ctx.params.explore_factor_schedule)
    else: #Default value inhereted from tensorpack
        schedule_str = "[[80, 2], [100, 3], [120, 4], [140, 5]]"
    logger.info("Setting explore factor schedule:{}".format(schedule_str))
    explore_factor_scheduler = ScheduledHyperParamSetter('explore_factor', json.loads(schedule_str))



    return TrainConfig(
        dataset=dataflow,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            learning_rate_scheduler, entropy_beta_scheduler, explore_factor_scheduler,
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            HumanHyperParamSetter('explore_factor'),
            NeputneHyperParamSetter('learning_rate', ctx),
            NeputneHyperParamSetter('entropy_beta', ctx),
            NeputneHyperParamSetter('explore_factor', ctx),
            master,
            StartProcOrThread(master),
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['logits'], neptuneLogger, HISTORY_LOGS), 1),
            neptuneLogger,
        ]),
        session_config=get_default_sess_config(0.5),
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
        max_epoch=num_epochs,
    )


HISTORY_LOGS = None

def run_training(ctx):
    """
    :type ctx: neptune.Context
    """
    global ENV_NAME, PREDICTOR_THREAD, EXPERIMENT_MODEL, HISTORY_LOGS, DEBUGING_INFO, FRAME_HISTORY

    ENV_NAME = ctx.params.env
    assert ENV_NAME

    DEBUGING_INFOING_INFO = hasattr(ctx.params, "debuging_info") and ctx.params.debuging_info == "True"

    # print "DEBUGGING INFO:{}".format(DEBUGING_INFO)
    FRAME_HISTORY = int(get_atribute(ctx, "frame_history", 4))

    # module_name, function_name = ctx.params.featureExtractor.split(".")
    module_name = ctx.params.experimentModelClass[:ctx.params.experimentModelClass.rfind('.')]
    class_name = ctx.params.experimentModelClass[ctx.params.experimentModelClass.rfind('.')+1:]
    experiment_model_class = importlib.import_module(module_name).__dict__[class_name]



    if hasattr(ctx.params, "stage"):
        # That's not the most elegant solution but well ;)
        stage = int(ctx.params.stage)
        EXPERIMENT_MODEL = experiment_model_class(ctx.params.experimentModelParameters, stage)
    else:
        EXPERIMENT_MODEL = experiment_model_class(ctx.params.experimentModelParameters)

    p = get_player();
    del p  # set NUM_ACTIONS
    EXPERIMENT_MODEL.set_number_of_actions(NUM_ACTIONS)


    if ctx.params.gpu:
        # print "CUDA_VISIBLE_DEVICES:{}".format(os.environ['CUDA_VISIBLE_DEVICES'])
        print "Set GPU:{}".format(ctx.params.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = ctx.params.gpu

    nr_gpu = get_nr_gpu()
    if nr_gpu > 1:
        predict_tower = range(nr_gpu)[-nr_gpu / 2:]
    else:
        predict_tower = [0]
    PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU

    if hasattr(ctx.params, "history_logs"):
        if ctx.params.history_logs == "":
            HISTORY_LOGS = ([], [], [])
        else:
            HISTORY_LOGS = json.loads(ctx.params.history_logs)


    config = get_config(ctx)

    if ctx.params.load != "":
        config.session_init = SaverRestore(ctx.params.load)

    if hasattr(ctx.params, "load_previous_stage"):
        if ctx.params.load_previous_stage != "":
            config.session_init = SaverRestore(ctx.params.load_previous_stage )


    config.tower = range(nr_gpu)[:-nr_gpu / 2] or [0]
    logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
        ','.join(map(str, config.tower)), ','.join(map(str, predict_tower))))
    AsyncMultiGPUTrainer(config, predict_tower=predict_tower).train()

    # For the moment this is a hack.
    # The neptune interface does not allow to return values

    if hasattr(ctx.params, "mother_experiment_id"):
        experiment_dir = ctx.dump_dir_url
        json_path = os.path.join(experiment_dir, ctx.params.mother_experiment_id + ".json")
        info_to_pass = {}
        info_to_pass["previous_experiment_id"] = ctx.job.id
        print "Experiment history logs to save:{}".format(HISTORY_LOGS)
        info_to_pass["history_logs"] = json.dumps(HISTORY_LOGS)
        with open(json_path, 'w') as outfile:
            json.dump(info_to_pass, outfile)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
#     parser.add_argument('--load', help='load model')
#     parser.add_argument('--env', help='env', required=True)
#     parser.add_argument('--task', help='task to perform',
#             choices=['play', 'eval', 'train'], default='train')
#     args = parser.parse_args()
#
#     ENV_NAME = args.env
#     assert ENV_NAME
#     p = get_player(); del p    # set NUM_ACTIONS
#
#     if args.gpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     if args.task != 'train':
#         assert args.load is not None
#
#     if args.task != 'train':
#         cfg = PredictConfig(
#                 model=Model(),
#                 session_init=SaverRestore(args.load),
#                 input_var_names=['state'],
#                 output_var_names=['logits'])
#         if args.task == 'play':
#             play_model(cfg)
#         elif args.task == 'eval':
#             eval_model_multithread(cfg, EVAL_EPISODE)
#     else:
#         nr_gpu = get_nr_gpu()
#         if nr_gpu > 1:
#             predict_tower = range(nr_gpu)[-nr_gpu/2:]
#         else:
#             predict_tower = [0]
#         PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
#         config = get_config()
#         if args.load:
#             config.session_init = SaverRestore(args.load)
#         config.tower = range(nr_gpu)[:-nr_gpu/2] or [0]
#         logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
#             ','.join(map(str, config.tower)), ','.join(map(str, predict_tower))))
#         AsyncMultiGPUTrainer(config, predict_tower=predict_tower).train()
