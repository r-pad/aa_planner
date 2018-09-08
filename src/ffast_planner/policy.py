#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Use a pre-trained policy to run actions on the robot.
"""

import numpy as np
import os
import six.moves.cPickle as cPickle


class Policy(object):
    """
    Run a forward pass of a pre-trained policy.
    """

    def __init__(self):
        """
        Get model from saved pickle.
        """
        dirpath = os.path.dirname(os.path.abspath(__file__))
        f = open(dirpath + '/model.save', 'rb')
        self.model = cPickle.load(f)
        f.close()


    def get_action(self, state):
        """
        Use saved trained policy and run a forward pass to get
        action (desired speed, steering angle).
        """
        return [0.6, np.random.random()-0.5]
