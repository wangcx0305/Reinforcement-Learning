#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:44:05 2017

@author: wangchunxiao
"""
import numpy as np
from mdp import MDP

class infMDP_policy_iter:
    
    @staticmethod
    def policy_eval(mdp, policy, beta):
        I = np.eye(mdp.nS)
        T = np.zeros((mdp.nS, mdp.nS))
        for s in range(mdp.nS):
            T[s,] = mdp.get_tprob(policy[s], s, )
        R = mdp.R
        V = np.dot(np.linalg.inv(I - beta * T), R)
        return V
    
    @staticmethod
    def improve_policy(mdp, policy, beta):
         V = infMDP_policy_iter.policy_eval(mdp, policy, beta)
         policy_new = np.zeros(mdp.nS, dtype = int)
         for s in range(mdp.nS):
             e = [np.dot(mdp.get_tprob(a, s,), V) for a in range(mdp.nA)]
             policy_new[s] = np.argmax(e)             
         return policy_new
         
    @staticmethod
    def policy_iter(mdp, beta):
        policy = np.random.randint(0, mdp.nA, mdp.nS)
        while not np.all(policy == infMDP_policy_iter.improve_policy(mdp, policy, beta)):
            policy = infMDP_policy_iter.improve_policy(mdp, policy, beta)     
        V = infMDP_policy_iter.policy_eval(mdp, policy, beta)
        return(policy, V)
    
        
        
        
        
        
        
