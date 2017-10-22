#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:11:23 2017

@author: wangchunxiao
"""

class Regret:
    
    def __init__(self, optimal_expected_reward):
    
        self.optimal_expected_reward = optimal_expected_reward
        self.expected_reward_best_arm = 0
        self.expected_cumulative_reward = 0
        self.n = 0

    def add(self, expected_reward_pulled_arm, expected_reward_best_arm):
        
        self.expected_reward_best_arm = expected_reward_best_arm
        self.expected_cumulative_reward += expected_reward_pulled_arm
        self.n += 1

    def get_cumulative_regret(self):
        
        return self.n * self.optimal_expected_reward - self.expected_cumulative_reward

    def get_simple_regret(self):
       
        return self.optimal_expected_reward - self.expected_reward_best_arm
