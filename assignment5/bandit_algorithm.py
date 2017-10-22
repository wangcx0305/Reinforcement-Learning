#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 07:05:10 2017

@author: wangchunxiao
"""

import numpy as np
import random

class BanditAlgorithm:

    def __init__(self, bandit):
        
        self.reset(bandit)

    def reset(self, bandit=None):
        
        if bandit is not None:
            self.bandit = bandit
        NumArms = bandit.get_NumArms()
        self.running_sum_rewards = np.zeros(NumArms, dtype = float)
        self.num_pulls = np.zeros(NumArms, dtype = int)
        self.total_pulls = 0
        self.average_rewards = np.zeros(NumArms, dtype = float)
        self.best_arm = random.randint(0, NumArms - 1)

    def pull(self):
        
        raise NotImplementedError

    def update(self, reward, arm):
        
        self.running_sum_rewards[arm] += reward
        self.num_pulls[arm] += 1
        self.total_pulls += 1

        self.average_rewards = np.divide(self.running_sum_rewards, self.num_pulls)
        self.average_rewards[np.isinf(self.average_rewards)] = 0
        self.average_rewards[np.isnan(self.average_rewards)] = 0

        self.best_arm = np.argmax(self.get_average_rewards())

    def get_bandit(self):
        
        return self.bandit

    def get_total_pulls(self):
        
        return self.total_pulls

    def get_average_rewards(self):
        
        return self.average_rewards

    def get_best_arm(self):
        
        return self.best_arm

    def get_name(self):
        
        raise NotImplementedError

class IncrementalUniform(BanditAlgorithm):
    
    def __init__(self, bandit):
        self.reset(bandit)

    def pull(self):
        
        arm = self.get_total_pulls() % self.bandit.get_NumArms()
        reward = self.bandit.pull(arm)

        self.update(reward, arm)

        return (arm, reward)

    def get_name(self):
       
        return "Incremental Uniform"

class UCB(BanditAlgorithm):
   
    def __init__(self, bandit):
        self.reset(bandit)

    def pull(self):
        
        if self.get_total_pulls() < self.bandit.get_NumArms():
            arm = self.get_total_pulls() % self.bandit.get_NumArms()
        else:
            averages = self.get_average_rewards()
            numerator = 2.*np.log(self.get_total_pulls())
            ratio = np.divide(numerator, self.num_pulls)
            exploration_term = np.sqrt(ratio)
            arm = np.argmax(averages + exploration_term)

        reward = self.bandit.pull(arm)

        self.update(reward, arm)

        return (arm, reward)

    def get_name(self):
        
        return "UCB"

class EpsilonGreedy(BanditAlgorithm):
    
    def __init__(self, bandit, epsilon=0.5):
        self.reset(bandit)
        self.epsilon = epsilon

    def pull(self):
        NumArms = self.bandit.get_NumArms()
        best_arm = self.get_best_arm()
        if random.uniform(0, 1) <= self.epsilon:
            arm = best_arm
        else:
            other_arms = list(set(range(NumArms)) - set([best_arm]))
            index = random.randint(0, NumArms - 2)
            arm = other_arms[index]

        reward = self.bandit.pull(arm)

        self.update(reward, arm)

        return (arm, reward)

    def get_name(self):
        
        return "Epsilon-Greedy, epsilon={0}".format(self.epsilon)
