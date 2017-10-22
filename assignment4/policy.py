#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:03:15 2017

@author: wangchunxiao
"""

import numpy as np

class policy:
    
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, state):
        return self.policy[state]

class random_policy(policy):
    
    def __init__(self, mdp, p = 0.5):
        
        self.mdp = mdp
        self.park_prob = p

    def get_action(self, state):
        (c, r, o, p) = self.mdp.get_counter_state(state)

        if p == 1:
            action = 2
        elif np.random.uniform() < self.park_prob:
            action = 0
        else:
            action = 1
        return action
        
class safe_park_policy(policy):
    
    def __init__(self, mdp, p = 0.5):
        self.mdp = mdp
        self.park_prob = p
        
    def get_action(self, state):
        (c, r, o, p) = self.mdp.get_counter_state(state)
        
        if p == 1:
            action = 2
        elif o == 1:
            action = 1
        elif np.random.uniform() < self.park_prob:
            action = 0
        else:
            action = 1
        return action

        
    