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
        r = 1.5
        x, y, yaw, x_dot, y_dot, yaw_dot = state

        # Transform state to relative space using convention 2
        y -= r
        dx = np.sqrt(np.square(x) + np.square(y)) - r
        theta = self._normalize_angle(np.arctan2(-x, y) + np.pi - yaw)
        ddx = x/(x**2 + y**2)**0.5*x_dot + y/(x**2 + y**2)**0.5*y_dot
        dtheta = x/(x**2 + y**2)*x_dot - y/(x**2 + y**2)*y_dot - yaw_dot
        newstate = np.array([dx, theta, ddx, dtheta])

        # Forward pass of policy network
        mean, log_std = [x[0] for x in self.model([newstate])]
        '''
        if mean[0] < 0.8:
            mean[0] = 0.8
        '''
        return mean


    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi).
        """
        angle = angle % (2*np.pi)
        if (angle > np.pi):
            angle -= 2*np.pi
        return angle
