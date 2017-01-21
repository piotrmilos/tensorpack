#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: run-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import shutil

import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import argparse
import six
import yaml

from tensorpack import *
from tensorpack.RL import *
from tensorpack.RL.common import MapPlayerState, PreventStuckPlayer, KeyboardPlayer
from tensorpack.RL.gymenv import GymEnv
from tensorpack.models.model_desc import InputVar
from tensorpack.predict.common import get_predict_func, PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
import multiprocessing as mp
from examples.OpenAIGym.train_atari_with_neptune import FRAME_HISTORY, DEBUGING_INFO

from easydict import EasyDict as edict
print "TF version:{}".format(tf.__version__)
IMAGE_SIZE = (84, 84)
# FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None
EXPERIMENT_MODEL = None

from common import play_one_episode

def get_player(dumpdir=None):
    pl = GymEnv(ENV_NAME, viz=0.001, dumpdir=dumpdir, force=True)

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
    pl = KeyboardPlayer(pl)

    return pl

# class Model(ModelDesc):
#     def _get_input_vars(self):
#         assert NUM_ACTIONS is not None
#         return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
#                 InputVar(tf.int32, (None,), 'action'),
#                 InputVar(tf.float32, (None,), 'futurereward') ]
#
#     def _get_NN_prediction(self, image):
#         image = image / 255.0
#         with argscope(Conv2D, nl=tf.nn.relu):
#             l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
#             l = MaxPooling('pool0', l, 2)
#             l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
#             l = MaxPooling('pool1', l, 2)
#             l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
#             l = MaxPooling('pool2', l, 2)
#             l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)
#
#         l = FullyConnected('fc0', l, 512, nl=tf.identity)
#         l = PReLU('prelu', l)
#         policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
#         return policy
#
#     def _build_graph(self, inputs):
#         state, action, futurereward = inputs
#         policy = self._get_NN_prediction(state)
#         self.logits = tf.nn.softmax(policy, name='logits')

def run_submission(cfg, dump_dir = 'gym-submit'):
    dirname = dump_dir
    player = get_player(dumpdir=dirname)
    # player = get_player("/home/piotr.milos")
    print "config:{}".format(cfg)
    predfunc = get_predict_func(cfg)

    for k in range(10):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def do_submit():
    dirname = 'gym-submit'
    gym.upload(dirname, api_key='xxx')


def run_atari_neptune_experiment(yamlFile = None, modelToLaod = None, epoch=None):
    global ENV_NAME, EXPERIMENT_MODEL, FRAME_HISTORY

    with open(yamlFile, 'r') as stream:
        try:
            yamlData =yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    argsDict = {}
    for v in yamlData["parameters"]:
        argsDict[v["name"]] = v["default"]

    args = edict(argsDict)
    ENV_NAME = args.env
    assert ENV_NAME

    if hasattr(args, "frame_history"):
        FRAME_HISTORY= args.frame_history
        # examples.OpenAIGym.train_atari_with_neptune.FRAME_HISTORY = args.frame_history
    else:
        FRAME_HISTORY = 4

    # FRAME_HISTORY = int(get_atribute(args, "frame_history", 4))
    logger.info("Environment Name: {}".format(ENV_NAME))

    # module_name, function_name = ctx.params.featureExtractor.split(".")
    module_name = args.experimentModelClass[:args.experimentModelClass.rfind('.')]
    class_name = args.experimentModelClass[args.experimentModelClass.rfind('.') + 1:]
    experiment_model_class = importlib.import_module(module_name).__dict__[class_name]
    EXPERIMENT_MODEL = experiment_model_class(args.experimentModelParameters)


    p = get_player();
    del p  # set NUM_ACTIONS. Bloody hack!
    EXPERIMENT_MODEL.set_number_of_actions(NUM_ACTIONS)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg = PredictConfig(
            model=EXPERIMENT_MODEL,
            session_init=SaverRestore(modelToLaod),
            input_var_names=['state'],
            output_var_names=['logits', 'pred_value'])
    dump_dir = os.path.join(dump_dir_root, str(epoch))
    print "Writing to:{}".format(dump_dir)
    run_submission(cfg, dump_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_yaml', help='The config file of the experiment') # nargs='*' in multi mode
    parser.add_argument('--experiment_neptune_id', help='The id of the ', required=True)
    parser.add_argument('--experiment_epoch', help="The epoch of the experiment", required=False)

    args = parser.parse_args()

    dump_dir_root = os.path.join("/tmp/video", args.experiment_yaml + args.experiment_neptune_id)

    # ROOT = "/home/piotr.milos/rl2"
    ROOT = "/mnt/storage_codi/piotr.milos/Projects/rl2"

    EXPERIMENTS_DIR = os.path.join(ROOT, "NeptuneIntegration", "Experiments")
    EXPERIMENTS_DUMPS = os.path.join(ROOT, "NeptuneIntegration", "Neptune_dumps")

    experiment_dir = os.path.join(EXPERIMENTS_DUMPS, args.experiment_yaml,
                                  "src", "train_log", "train_atari_with_neptune_proxy" + args.experiment_neptune_id)
    print "Experiment path:{}".format(experiment_dir)
    files = os.listdir(experiment_dir)

    yamlFile = os.path.join(EXPERIMENTS_DIR, args.experiment_yaml + ".yaml")

    modelList = []
    for file_name in files:
        if "model" in file_name:
            modelList.append(file_name)

    # modelList = sorted(modelList)
    availableEpochs = []
    chooseFileDict = {}
    for x in modelList:
        stepNumber = int(x[6:])
        epochNumber = stepNumber / 6000
        availableEpochs.append(epochNumber)
        chooseFileDict[epochNumber] = x

    if args.experiment_epoch != None:
        if args.experiment_epoch == "ALL":
            epochToBeTested = sorted(availableEpochs)
        else:
            ett = int(args.experiment_epoch)
            if ett not in availableEpochs:
                logger.info("Specified epoch {} is not available. Skipping.".format(ett))
                sys.exit(0)
            epochToBeTested = [ett]
    else:
        availableEpochs = sorted(availableEpochs)
        epochToBeTested = [raw_input("Choose the epoch to test {}:".format(availableEpochs))]

    for e in epochToBeTested:
        logger.info("Testing epoch:{}".format(e))
        print "Testing epoch:{}".format(e)
        modelToLaod = os.path.join(experiment_dir, chooseFileDict[int(e)])
        print "Model to load:{}".format(modelToLaod)
        shutil.copy(modelToLaod, "/tmp/model")
        modelToLaod = "/tmp/model"
        run_atari_neptune_experiment(yamlFile=yamlFile, modelToLaod=modelToLaod)
        # kwargs = {"yamlFile":yamlFile, "modelToLaod":modelToLaod, "epoch":e}

        # p = mp.Process(target=run_atari_neptune_experiment, kwargs=kwargs)
        # p.start()
        # p.join()
    # run_atari_neptune_experiment(yamlFile=yamlFile, modelToLaod=modelToLaod)


