#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import pip
installed_packages = pip.get_installed_distributions()

import random, time
import threading, multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue

from tensorpack import *
from tensorpack.callbacks import neptune
from tensorpack.callbacks.base import Callback
from tensorpack.predict import get_predict_func
from tensorpack.utils.concurrency import *
from tensorpack.utils.stats import  *
from tensorpack.callbacks import *

global get_player
get_player = None

def play_one_episode(player, func, verbose=False):
    def f(s):
        spc = player.get_action_space()
        act = func([[s]])[0][0].argmax()
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act
    return np.mean(player.play_one_episode(f))

def play_model(cfg):
    player = get_player(viz=0.01)
    predfunc = get_predict_func(cfg)
    while True:
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def eval_with_funcs(predict_funcs, nr_eval):
    class Worker(StoppableThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            player = get_player(train=False)
            while not self.stopped():
                try:
                    score = play_one_episode(player, self.func)
                    #print "Score, ", score
                except RuntimeError:
                    return
                self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predict_funcs]

    for k in threads:
        k.start()
        time.sleep(0.1) # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            r = q.get()
            stat.feed(r)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads: k.stop()
        for k in threads: k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        if stat.count > 0:
            # return (stat.average, stat.max, stat.std)
            return (stat.average, stat.max, 0)
        return (0, 0, 0)

def eval_model_multithread(cfg, nr_eval):
    func = get_predict_func(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))

class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, neptune_logger = None, history_logs = None):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.neptuneLogger = neptune_logger
        if self.neptuneLogger != None:
            self.neptuneLogger.add_channels(["mean_score", "max_score", "std_of_score"])
        self.history_logs = history_logs
        self.history_emitted = False

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predict_func(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger_epoch(self):
        t = time.time()
        mean, max, std = eval_with_funcs(self.pred_funcs, nr_eval=self.eval_episode)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
            print "PM:Something strage is invoked. Be careful ;)"
        self.trainer.write_scalar_summary('mean_score', mean)
        self.trainer.write_scalar_summary('max_score', max)
        self.trainer.write_scalar_summary('std_of_score', std)


        if self.neptuneLogger != None:
            if not self.history_emitted and self.history_logs != None:
                for num, val in enumerate(self.history_logs[0]):
                    self.neptuneLogger.sendMassage("mean_score", x=num, y=val)
                for num, val in enumerate(self.history_logs[1]):
                    self.neptuneLogger.sendMassage("max_score", x=num, y=val)
                for num, val in enumerate(self.history_logs[2]):
                    self.neptuneLogger.sendMassage("std_of_score", x=num, y=val)

                    self.history_emitted = True

            if self.history_logs == None:
                self.neptuneLogger.sendMassage("mean_score", x=self.epoch_num, y=mean)
                self.neptuneLogger.sendMassage("max_score", x=self.epoch_num, y=max)
                self.neptuneLogger.sendMassage("std_of_score", x=self.epoch_num, y=std)
            else:
                self.history_logs[0].append(mean)
                self.history_logs[1].append(max)
                self.history_logs[2].append(std)
                print "HIStorY LOGS in progress:{}".format(self.history_logs[0])
                self.neptuneLogger.sendMassage("mean_score", x=len(self.history_logs[0]) - 1, y=mean)
                self.neptuneLogger.sendMassage("max_score", x=len(self.history_logs[1]) - 1, y=max)
                self.neptuneLogger.sendMassage("std_of_score", x=len(self.history_logs[2]) - 1, y=std)



