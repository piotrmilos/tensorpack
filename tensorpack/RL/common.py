#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import numpy as np
from collections import deque

import readchar as readchar

from .envbase import ProxyPlayer

__all__ = ['PreventStuckPlayer', 'LimitLengthPlayer', 'AutoRestartPlayer',
        'MapPlayerState', 'KeyboardPlayer']

class PreventStuckPlayer(ProxyPlayer):
    """ Prevent the player from getting stuck (repeating a no-op)
    by inserting a different action. Useful in games such as Atari Breakout
    where the agent needs to press the 'start' button to start playing.
    """
    # TODO hash the state as well?
    def __init__(self, player, nr_repeat, action):
        """
        :param nr_repeat: trigger the 'action' after this many of repeated action
        :param action: the action to be triggered to get out of stuck
        Does auto-reset, but doesn't auto-restart the underlying player.
        """
        super(PreventStuckPlayer, self).__init__(player)
        self.act_que = deque(maxlen=nr_repeat)
        self.trigger_action = action

    def action(self, act):
        self.act_que.append(act)
        if self.act_que.count(self.act_que[0]) == self.act_que.maxlen:
            act = self.trigger_action
        r, isOver = self.player.action(act)
        if isOver:
            self.act_que.clear()
        return (r, isOver)

    def restart_episode(self):
        super(PreventStuckPlayer, self).restart_episode()
        self.act_que.clear()

class LimitLengthPlayer(ProxyPlayer):
    """ Limit the total number of actions in an episode.
        Will auto restart the underlying player on timeout
    """
    def __init__(self, player, limit):
        super(LimitLengthPlayer, self).__init__(player)
        self.limit = limit
        self.cnt = 0

    def action(self, act):
        r, isOver = self.player.action(act)
        self.cnt += 1
        if self.cnt >= self.limit:
            isOver = True
            self.finish_episode()
            self.restart_episode()
        if isOver:
            self.cnt = 0
        return (r, isOver)

    def restart_episode(self):
        self.player.restart_episode()
        self.cnt = 0

class AutoRestartPlayer(ProxyPlayer):
    """ Auto-restart the player on episode ends,
        in case some player wasn't designed to do so. """
    def action(self, act):
        r, isOver = self.player.action(act)
        if isOver:
            self.player.finish_episode()
            self.player.restart_episode()
        return r, isOver

class MapPlayerState(ProxyPlayer):
    def __init__(self, player, func):
        super(MapPlayerState, self).__init__(player)
        self.func = func

    def current_state(self):
        return self.func(self.player.current_state())

class KeyboardPlayer(ProxyPlayer):

    def __init__(self, player):
        self.player = player

        self.ACTION_MEANING = {
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "RIGHT",
            4: "LEFT",
            5: "DOWN",
            6: "UPRIGHT",
            7: "UPLEFT",
            8: "DOWNRIGHT",
            9: "DOWNLEFT",
            10: "UPFIRE",
            11: "RIGHTFIRE",
            12: "LEFTFIRE",
            13: "DOWNFIRE",
            14: "UPRIGHTFIRE",
            15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE",
            17: "DOWNLEFTFIRE",
        }

        self.map = {v: k for k, v in self.ACTION_MEANING.iteritems()}
        self.mode = 1

    def map_input(self, c):
        if c == 'w':
            return self.map["UP"]
        if c == 's':
            return self.map["DOWN"]
        if c == 'a':
            return self.map["LEFT"]
        if c == 'd':
            return self.map["RIGHT"]
        if c == 'q':
            return self.map["UPLEFT"]
        if c == 'e':
            return self.map["UPRIGHT"]

        if c == 't':
            return self.map["UPFIRE"]
        if c == 'g':
            return self.map["DOWNFIRE"]
        if c == 'f':
            return self.map["LEFTFIRE"]
        if c == 'h':
            return self.map["RIGHTFIRE"]
        if c == 'r':
            return self.map["UPLEFTFIRE"]
        if c == 'y':
            return self.map["UPRIGHTFIRE"]


        print "Shoud not happen!"
        return self.map["NOOP"]

    def action(self, act):
        # print "Entering with action:{}".format(self.ACTION_MEANING[act])
        old_act = act
        if self.mode == 0:
            name = raw_input("Enter command: ")
            print "Command:{}".format(name)
            if "[" in name:
                self.mode = 1
            if "ex" in name:
                file_name = name.split(" ")[1]
                with open(file_name) as f:
                    commands = f.readline()
                commands = commands.split(" ")
                self.commands = []
                for c in commands:
                    if "*" in c:
                        cmd = c.split("*")[0]
                        counter = int(c.split("*")[1])
                        self.commands += ([cmd] * counter)
                    else:
                        self.commands += [c]
                print "Commands:{}".format(self.commands)
                self.mode = 3
                self.counter = 0
            act = self.map["NOOP"]

        if self.mode == 1:
            c = readchar.readkey()
            # print "Key:{}".format(c)
            change_action = True
            if c == '[':
                self.mode = 0
                act = self.map["NOOP"]
                change_action = False
            if c == ' ':
                change_action = False
            if c == '.':
                raise ZeroDivisionError

            if change_action:
                act = self.map_input(c)

        if self.mode == 3:
            if self.commands[self.counter] in  self.map:
                act = self.map[self.commands[self.counter]]
            else:
                act = 0
            self.counter += 1
            if self.counter == len(self.commands):
                self.mode = 1

        print "Policy action:{} our action {}".format(self.ACTION_MEANING[old_act],
                                                      self.ACTION_MEANING[act])

        r, isOver = self.player.action(act)
        return r, isOver

