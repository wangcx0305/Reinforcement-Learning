#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:26:09 2017

@author: wangchunxiao
"""

import numpy as np
import random

class Bandit:
    def __init__(self, NumArms, name='bandit'):
        
        self.NumArms = NumArms
        self.name = name

    def get_NumArms(self):
        
        return self.NumArms

    def pull(self, arm):
        
        raise NotImplementedError

    def get_expected_reward_optimal_arm(self):
        
        raise NotImplementedError

    def get_expected_reward_arm(self, arm):
       
        raise NotImplementedError

    def get_name(self):
        
        return self.name

class SBRDBandit(Bandit):
    
    def __init__(self, arm_params, name='bandit'):
        
        for i, (r, p) in enumerate(arm_params):
            if r < 0. or r > 1.:
                raise ValueError('Invalid r param ({0}) for arm {1}'.format(r, i))
            if p < 0. or p > 1.:
                raise ValueError('Invalid p param ({0}) for arm {1}'.format(p, i))

        Bandit.__init__(self, len(arm_params), name)
        self.arm_params = arm_params
        self.optimal_arm = np.argmax([p*r for (r, p) in arm_params])

    def pull(self, arm):
        
        r, p = self.arm_params[arm]
        success = random.uniform(0, 1) <= p
        return r if success else 0

    def get_expected_reward_optimal_arm(self):
        
        return self.get_expected_reward_arm(self.optimal_arm)

    def get_expected_reward_arm(self, arm):
        
        (r, p) = self.arm_params[arm]
        return r * p
