#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 02:15:30 2017

@author: wangchunxiao
"""
import numpy as np
import random

from general_simulator import General_Simulator

class Q_Learner:
    
    def __init__(self, mdp, initial_state, epsilon = 0.1, alpha = 0.01):

        self.simulator = General_Simulator(mdp, initials = initial_state)
        self.initial_state = initial_state
        
        self.qtable = np.zeros((mdp.nS, mdp.nA))
        self.num_learning_trials = 0

        self.alpha = alpha 
        self.beta = 0.9

        self.epsilon = epsilon # epsilon-greedy: probability select random action
        
        self.n_trial = 100

    def learner_measure_trial(self):
       
        total_reward = 0
        sseq = []
        aseq = []
        step = 0
        self.simulator.curs = self.initial_state
        self.simulator.terminal_state = False
        while not self.simulator.terminal_state and step < self.n_trial:
            curs = self.simulator.curs
            sseq.append(curs)
            reward = self.simulator.mdp.get_reward(curs)
            total_reward += reward
            
            action = np.argmax(self.qtable[curs,:])
            aseq.append(action)

            self.simulator.transit(action)

            step += 1

        return (total_reward, sseq, aseq)

    def learning_trial(self):
        
        sseq = []
        rewseq = []
        aseq = []
        step = 0
        self.simulator.curs = self.initial_state
        self.simulator.terminal_state = False
        while not self.simulator.terminal_state and step < self.n_trial:
            curs = self.simulator.curs
            sseq.append(curs)
            reward = self.simulator.mdp.get_reward(curs)
            rewseq.append(reward)

            action = self.explore_exploit_policy(curs)
            aseq.append(action)
            self.simulator.transit(action)

            step += 1

        curs = self.simulator.curs
        sseq.append(curs)
        reward = self.simulator.mdp.get_reward(curs)
        rewseq.append(reward)

        self.reverse_updates(sseq, rewseq, aseq)
        self.num_learning_trials += 1

    def explore_exploit_policy(self, curs):
        
        if self.num_learning_trials == 0 or random.uniform(0, 1) <= self.epsilon:
            n_action = self.qtable.shape[1]
            return random.randint(0, n_action - 1)
        else:
            return np.argmax(self.qtable[curs,:])

    def reverse_updates(self, sseq, rewseq, aseq):
        
        #assert(len(sseq) == len(aseq) + 1)
        #assert(len(sseq) == len(rewseq))

        for i in reversed(range(len(aseq))):
            curs = sseq[i]
            action = aseq[i]
            nexts = sseq[i+1]
            reward = rewseq[i]
            self.qtable[curs, action] = self.compute_update(curs, action, nexts, reward)

    def compute_update(self, pres, prea, curs, reward):
        n_action = self.qtable.shape[1]
        prevalue = self.qtable[pres, prea]
        optimal_cur_estimate = np.max([self.qtable[curs, a] for a in range(n_action)])
        learn_value = prevalue + self.alpha * (reward + self.beta * optimal_cur_estimate - prevalue)
        return learn_value

