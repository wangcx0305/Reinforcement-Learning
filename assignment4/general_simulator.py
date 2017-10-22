#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:55:18 2017

@author: wangchunxiao
"""

import numpy as np
from mdp import MDP

class General_Simulator:
    def __init__(self, mdp, initials=0):
        self.mdp = mdp
        self.curs = initials
        self.terminal_state = False


    def policy_measure(self, policy):
        total_reward = 0
        sseq = []
        aseq = []
        while not self.terminal_state:
            sseq.append(self.curs)
            reward = self.mdp.get_reward(self.curs)
            total_reward += reward
            action = policy.get_action(self.curs)
            aseq.append(action)
            
            self.transit(action)
            
        return (total_reward, sseq, aseq)


    def transit(self, action):
        if self.terminal_state:
            return
        states = np.array(range(self.mdp.nS))
        probs = self.mdp.get_tprob(action, self.curs)
        if np.sum(probs) == 0:
            return
        bins = np.add.accumulate(probs)
        nexts = states[np.digitize(np.random.random_sample(1), bins)[0]]
        self.curs = nexts
        self.terminal_state = True
        for a in range(self.mdp.nA):
            probs = self.mdp.get_tprob(a, self.curs)
            if np.sum(probs) > 0:
                self.terminal_state = False
                break

        
        
