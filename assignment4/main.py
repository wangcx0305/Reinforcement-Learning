#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:30:33 2017

@author: wangchunxiao
"""
import numpy as np
from mdp import MDP
from parking_MDP_creation import P_MDP
from policy import policy, random_policy, safe_park_policy
from general_simulator import General_Simulator
from mdp_opt import infMDP_policy_iter
from Q_learning import Q_Learner
from matplotlib import pyplot as plt


######PART 1########

mdp1 = P_MDP(10, -1000, -10000, -1, 10)

with open('P_MDP1.txt', 'w') as f:
    f.write('{0} {1}\n\n'.format(mdp1.nS, mdp1.nA))
    for a in range(mdp1.nA):
        transition = '\n'.join(' '.join('{0:0.8f}'.format(elem) for elem in row) for row in mdp1.T[a])
        f.write('{0}\n\n'.format(transition))
    f.write(' '.join('{0:0.8f}'.format(R) for R in mdp1.R))
    f.write('\n')
f.close()    
    

mdp2 = P_MDP(10, -1000, -10000, -10, 10)

with open('P_MDP2.txt', 'w') as f:
    f.write('{0} {1}\n\n'.format(mdp2.nS, mdp2.nA))
    for a in range(mdp2.nA):
        transition = '\n'.join(' '.join('{0:0.8f}'.format(elem) for elem in row) for row in mdp2.T[a])
        f.write('{0}\n\n'.format(transition))
    f.write(' '.join('{0:0.8f}'.format(R) for R in mdp2.R))
    f.write('\n') 
f.close()

#########PART II########

########MDP 1###########

policy1 = random_policy(mdp1, 0.1)
reward = 0
for i in range(1000):
    simulator = General_Simulator(mdp1, 0)
    (total_reward, state_seq, action_seq) = simulator.policy_measure(policy1)
    reward = reward + total_reward
avr_reward = reward / 1000.0 #-4032.956
avr_reward

policy2 = safe_park_policy(mdp1, 0.1)
reward = 0
for i in range(1000):
    simulator = General_Simulator(mdp1, 0)
    (total_reward, state_seq, action_seq) = simulator.policy_measure(policy2)
    reward = reward + total_reward
avr_reward = reward / 1000.0 #-3614.834
avr_reward


optimal_policy, _ = infMDP_policy_iter.policy_iter(mdp1, 0.9)
policy3 = policy(optimal_policy)
reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator = General_Simulator(mdp1, 0)
    (total_reward, state_seq, action_seq) = simulator.policy_measure(policy3)
    reward = reward + total_reward
avr_reward = reward / 1000.0 #86.538
avr_reward

########MDP 2###########

policy4 = random_policy(mdp2, 0.1)
reward = 0
for i in range(1000):
    simulator = General_Simulator(mdp2, 0)
    (total_reward, state_seq, action_seq) = simulator.policy_measure(policy4)
    reward = reward + total_reward
avr_reward = reward / 1000.0 #-4237.840
avr_reward

policy5 = safe_park_policy(mdp2, 0.1)
reward = 0
for i in range(1000):
    simulator = General_Simulator(mdp1, 0)
    (total_reward, state_seq, action_seq) = simulator.policy_measure(policy5)
    reward = reward + total_reward
avr_reward = reward / 1000.0 #-3917.325
avr_reward

optimal_policy, _ = infMDP_policy_iter.policy_iter(mdp2, 0.9)
policy6 = policy(optimal_policy)
reward = 0
num_trials = 1000
for i in range(num_trials):
    simulator = General_Simulator(mdp2, 0)
    (total_reward, state_seq, action_seq) = simulator.policy_measure(policy6)
    reward = reward + total_reward
avr_reward = reward / 1000.0 #58.96
avr_reward


######PART III##############
n_learning_trials = 100
n_simulate_trials = 1000
n_learning_seg = 20


######MDP1 epsilon evaluation#####
lepsilon = [0.1, 0.3, 0.5, 0.7, 0.9]
learning_rate = 0.01
lavgre = np.zeros([5, n_learning_seg])

for e, epsilon in enumerate(lepsilon):

    qlearner = Q_Learner(mdp1, 0, epsilon = epsilon, alpha=learning_rate)
    
    for seg in range(n_learning_seg):
        for trial in range(n_learning_trials):
            qlearner.learning_trial()

        reward = 0
        for trial in range(n_simulate_trials):
            (total_reward, sseq, aseq) = qlearner.learner_measure_trial()
            reward += total_reward
        avg_reward = float(reward / n_simulate_trials)
        lavgre[e, seg] = avg_reward

k = range(20)
plt.plot(k, lavgre[0, ])
plt.plot(k, lavgre[1, ])
plt.plot(k, lavgre[2, ])
plt.plot(k, lavgre[3, ])
plt.plot(k, lavgre[4, ])
plt.xlabel('segment number')
plt.ylabel('average reward')
plt.title(r'different $\epsilon$ for MDP1')
plt.legend([r'$\epsilon = 0.1$', r'$\epsilon = 0.3$', r'$\epsilon = 0.5$',  r'$\epsilon = 0.7$', r'$\epsilon = 0.9$'], loc = 'lower right')

##########MDP1 alpha evaluation######
llearning_rate = [0.001, 0.01, 0.1, 0.5]
epsilon = 0.1
lavgre = np.zeros([4, n_learning_seg])

for e, learning_rate in enumerate(llearning_rate):

    qlearner = Q_Learner(mdp1, 0, epsilon = epsilon, alpha=learning_rate)
    
    for seg in range(n_learning_seg):
        for trial in range(n_learning_trials):
            qlearner.learning_trial()

        reward = 0
        for trial in range(n_simulate_trials):
            (total_reward, sseq, aseq) = qlearner.learner_measure_trial()
            reward += total_reward
        avg_reward = float(reward / n_simulate_trials)
        lavgre[e, seg] = avg_reward

k = range(20)
plt.plot(k, lavgre[0, ])
plt.plot(k, lavgre[1, ])
plt.plot(k, lavgre[2, ])
plt.plot(k, lavgre[3, ])
plt.xlabel('segment number')
plt.ylabel('average reward')
plt.title(r'different $\alpha$ for MDP1')
plt.legend([r'$\alpha = 0.001$', r'$\alpha = 0.01$', r'$\alpha = 0.1$',  r'$\alpha = 0.5$'], loc = 'lower right')


#######MDP2 epsilon evaluation######
lepsilon = [0.1, 0.3, 0.5, 0.7, 0.9]
learning_rate = 0.01
lavgre = np.zeros([5, n_learning_seg])

for e, epsilon in enumerate(lepsilon):

    qlearner = Q_Learner(mdp2, 0, epsilon = epsilon, alpha=learning_rate)
    
    for seg in range(n_learning_seg):
        for trial in range(n_learning_trials):
            qlearner.learning_trial()

        reward = 0
        for trial in range(n_simulate_trials):
            (total_reward, sseq, aseq) = qlearner.learner_measure_trial()
            reward += total_reward
        avg_reward = float(reward / n_simulate_trials)
        lavgre[e, seg] = avg_reward

k = range(20)
plt.plot(k, lavgre[0, ])
plt.plot(k, lavgre[1, ])
plt.plot(k, lavgre[2, ])
plt.plot(k, lavgre[3, ])
plt.plot(k, lavgre[4, ])
plt.xlabel('segment number')
plt.ylabel('average reward')
plt.title(r'different $\epsilon$ for MDP2')
plt.legend([r'$\epsilon = 0.1$', r'$\epsilon = 0.3$', r'$\epsilon = 0.5$',  r'$\epsilon = 0.7$', r'$\epsilon = 0.9$'], loc = 'lower right')


##########MDP2 alpha evaluation######
llearning_rate = [0.001, 0.01, 0.1, 0.5]
epsilon = 0.1
lavgre = np.zeros([4, n_learning_seg])

for e, learning_rate in enumerate(llearning_rate):

    qlearner = Q_Learner(mdp2, 0, epsilon = epsilon, alpha=learning_rate)
    
    for seg in range(n_learning_seg):
        for trial in range(n_learning_trials):
            qlearner.learning_trial()

        reward = 0
        for trial in range(n_simulate_trials):
            (total_reward, sseq, aseq) = qlearner.learner_measure_trial()
            reward += total_reward
        avg_reward = float(reward / n_simulate_trials)
        lavgre[e, seg] = avg_reward

k = range(20)
plt.plot(k, lavgre[0, ])
plt.plot(k, lavgre[1, ])
plt.plot(k, lavgre[2, ])
plt.plot(k, lavgre[3, ])
plt.xlabel('segment number')
plt.ylabel('average reward')
plt.title(r'different $\alpha$ for MDP2')
plt.legend([r'$\alpha = 0.001$', r'$\alpha = 0.01$', r'$\alpha = 0.1$',  r'$\alpha = 0.5$'], loc = 'lower right')





















    
