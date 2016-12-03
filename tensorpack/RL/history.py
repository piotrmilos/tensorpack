#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: history.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from collections import deque
from .envbase import ProxyPlayer

__all__ = ['HistoryFramePlayer']

class HistoryFramePlayer(ProxyPlayer):
    """ Include history frames in state, or use black images
        Assume player will do auto-restart.
    """
    def __init__(self, player, hist_len, history_processor = None):
        """
        :param hist_len: total length of the state, including the current
            and `hist_len-1` history
        """
        super(HistoryFramePlayer, self).__init__(player)
        self.history = deque(maxlen=hist_len)

        s = self.player.current_state()
        self.history.append(s)
        self.concatenate_axis = len(s.shape) -1
        self.history_processor = history_processor


    def _current_state(self):
        assert len(self.history) != 0
        diff_len = self.history.maxlen - len(self.history)
        if diff_len == 0:
            return np.concatenate(self.history, axis=self.concatenate_axis)
        zeros = [np.zeros_like(self.history[0]) for k in range(diff_len)]
        for k in self.history:
            zeros.append(k)
        assert len(zeros) == self.history.maxlen
        return np.concatenate(zeros, axis=self.concatenate_axis)

    def current_state(self):
        state = self._current_state()
        if self.history_processor != None:
            return self.history_processor(state)
        else:
            return state

    def action(self, act):
        r, isOver = self.player.action(act)
        s = self.player.current_state()
        self.history.append(s)

        if isOver:  # s would be a new episode
            self.history.clear()
            self.history.append(s)
        return (r, isOver)

    def restart_episode(self):
        super(HistoryFramePlayer, self).restart_episode()
        self.history.clear()
        self.history.append(self.player.current_state())

