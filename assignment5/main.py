#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:42:39 2017

@author: wangchunxiao
"""
from Plot import Plot
from regret import Regret
import numpy as np
import time

############PART 1################

from bandit import SBRDBandit
from bandit_algorithm import IncrementalUniform
from bandit_algorithm import UCB
from bandit_algorithm import EpsilonGreedy


############PART 2#################

# bandit problem 1
arm_params = []
for a in range(9):
    arm_params.append((0.05, 1))
arm_params.append((1, 0.1))
bandit1 = SBRDBandit(arm_params, 'bandit1')

# bandit problem 2
arm_params = []
for a in range(20):
    arm_params.append((1.*(a+1)/20, 0.1))
bandit2 = SBRDBandit(arm_params, 'bandit2')

# self_defined bandit problem
arm_params = []
for a in range(10):
    arm_params.append((0.01, 1))
for a in range(9):
    arm_params.append((0.9, 0.5))
arm_params.append((1, 0.5))
bandit3 = SBRDBandit(arm_params, 'sel_defined_bandit')


############PARTS 3 and 4###############

num_pulls = 1000000
num_trials = 20

def run_bandit_experiment(bandit, num_pulls, num_trials):

    algorithm1 = IncrementalUniform(bandit)
    algorithm2 = UCB(bandit)
    algorithm3 = EpsilonGreedy(bandit)
    algorithms = [algorithm1, algorithm2, algorithm3]

    plot_sample_rate = 1
    plot = Plot(num_pulls, num_trials,
        [a.get_name() for a in algorithms], plot_sample_rate)

    for a in algorithms:
        print ('\nRunning algorithm {0}...'.format(a.get_name()))
        plot.reset_trial()

        best_arms = np.zeros(num_trials)
        for t in range(num_trials):
            print ('Running trial {0}...'.format(t))
            start = time.time()

            plot.begin_trial()
            optimal_expected_reward = bandit.get_expected_reward_optimal_arm()
            regret = Regret(optimal_expected_reward)
            a.reset(bandit)

            for i in range(num_pulls):
                
                pulled_arm, _ = a.pull()

                best_arm = a.get_best_arm()
                expected_reward_pulled_arm = bandit.get_expected_reward_arm(pulled_arm)
                expected_reward_best_arm = bandit.get_expected_reward_arm(best_arm)
                regret.add(expected_reward_pulled_arm, expected_reward_best_arm)

                if i % plot_sample_rate == 0:
                    plot.add_point(i, regret.get_simple_regret(),
                        regret.get_cumulative_regret(), a.get_name())

            end = time.time()
            print ('\telapsed: {0}'.format(end-start))
            print ('\tbest arm: {0}'.format(a.get_best_arm()))
            best_arms[t] = a.get_best_arm()

        print ("Best arm distribution: ")
        print (np.histogram(best_arms, bins=range(21)))

    plot.plot_simple_regret(bandit.get_name())
    plot.plot_cumulative_regret(bandit.get_name())

    plot.save('{0}_data'.format(bandit.get_name()))

run_bandit_experiment(bandit1, num_pulls, num_trials)
run_bandit_experiment(bandit2, num_pulls, num_trials)
run_bandit_experiment(bandit3, num_pulls, num_trials)
