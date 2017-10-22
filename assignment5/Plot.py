#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 19:31:28 2017

@author: wangchunxiao
"""

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Plot:
    
    def __init__(self, num_arm_pulls, num_trials, algorithms, sample_rate = 100):
        
        num_points = (int(num_arm_pulls)/int(sample_rate))
        self.arm_pulls = {}
        self.simple_regret = {}
        self.cumulative_regret = {}
        for alg in algorithms:
            self.arm_pulls[alg] = np.zeros((num_points, num_trials))
            self.simple_regret[alg] = np.zeros((num_points, num_trials))
            self.cumulative_regret[alg] = np.zeros((num_points, num_trials))

        self.reset_trial()

    def save(self, output_file):
        
        data = (self.arm_pulls, self.simple_regret, self.cumulative_regret)
        np.savez(output_file, data=data)

    def load(self, input_file):
        
        content = np.load(input_file)
        (self.arm_pulls, self.simple_regret, self.cumulative_regret) = content['data']
        self.reset_trial()

    def reset_trial(self):
        
        self.arm_pulls_counter = 0
        self.trial_counter = -1

    def begin_trial(self):
        
        self.arm_pulls_counter = 0
        self.trial_counter += 1

    def add_point(self, num_arm_pulls, simple_regret, cumulative_regret, algorithm):
       
        self.arm_pulls[algorithm][self.arm_pulls_counter, self.trial_counter] = num_arm_pulls
        self.simple_regret[algorithm][self.arm_pulls_counter, self.trial_counter] = simple_regret
        self.cumulative_regret[algorithm][self.arm_pulls_counter, self.trial_counter] = cumulative_regret
        self.arm_pulls_counter += 1

    def plot_simple_regret(self, experiment_name, sample_rate = 1, end_index = None):
        
        self._plot('Simple Regret', self.arm_pulls, self.simple_regret,
            '{0}_simple_regret.png'.format(experiment_name), sample_rate, end_index)

    def plot_cumulative_regret(self, experiment_name, sample_rate = 1, end_index = None):
       
        self._plot('Cumulative Regret', self.arm_pulls, self.cumulative_regret,
            '{0}_cumulative_regret.png'.format(experiment_name), sample_rate, end_index)

    def _plot(self, regret_type, x, y, output_file, sample_rate, end_index):
       
        color_list = ['black', 'red', 'green']
        for i, algorithm in enumerate(x):
            x_data = x[algorithm][0::sample_rate,:] if end_index is None else x[algorithm][0:end_index:sample_rate,:]
            y_data = y[algorithm][0::sample_rate,:] if end_index is None else y[algorithm][0:end_index:sample_rate,:]
            plt.plot(np.mean(x_data, axis=1), np.mean(y_data, axis = 1),
                color=color_list[i], linewidth = 1.5,
                linestyle = '-', label = algorithm)
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('NumArms')
        plt.ylabel(regret_type)
        plt.title('{0} vs. NumArms'.format(regret_type))

        plt.savefig(output_file, dpi=72)
        plt.clf()
