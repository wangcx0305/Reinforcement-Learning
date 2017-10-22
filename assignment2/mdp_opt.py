#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:44:05 2017

@author: wangchunxiao
"""
import numpy as np
from mdp import MDP

class MDP_OPT:
    
    @staticmethod
    def finiteH_OPT(mdp, H):
        V = np.zeros((mdp.nS, H + 1))
        Policy = np.zeros((mdp.nS, H))
        
        for s in range(mdp.nS):
           V[s, 0] = mdp.get_reward(s)
           
        for h in range(H):
            for s in range(mdp.nS):
                e = [np.dot(mdp.get_tprob(a, s,), V[:, h]) for a in range(mdp.nA)]
                V[s, h + 1] = mdp.get_reward(s) + np.max(e)
                Policy[s, h] = np.argmax(e)
                
        return (V, Policy)
        
