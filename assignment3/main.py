#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:29:05 2017

@author: wangchunxiao
"""
from mdp import MDP
from mdp_opt import infMDP_policy_iter
from parking_MDP_creation import P_MDP
import numpy as np

################problem 2###################################
mdp = MDP()
mdp.loadfile('MDP1-hw3.txt')
policy_01, V_01 = infMDP_policy_iter.policy_iter(mdp, 0.1)
policy_09, V_09 = infMDP_policy_iter.policy_iter(mdp, 0.9)
np.savetxt('policy_MDP1_beta01.txt', policy_01, fmt = '%d')
np.savetxt('value_MDP1_beta01.txt', V_01, fmt = '%.4f')
np.savetxt('policy_MDP1_beta09.txt', policy_09, fmt = '%d')
np.savetxt('value_MDP1_beta09.txt', V_09, fmt = '%.4f')

mdp = MDP()
mdp.loadfile('MDP2-hw3.txt')
policy_01, V_01 = infMDP_policy_iter.policy_iter(mdp, 0.1)
policy_09, V_09 = infMDP_policy_iter.policy_iter(mdp, 0.9)
np.savetxt('policy_MDP2_beta01.txt', policy_01, fmt = '%d')
np.savetxt('value_MDP2_beta01.txt', V_01, fmt = '%.4f')
np.savetxt('policy_MDP2_beta09.txt', policy_09, fmt = '%d')
np.savetxt('value_MDP2_beta09.txt', V_09, fmt = '%.4f')


####################problem 3###############################
def counter_context(counter, mdp):
    state = mdp.counter_state[counter]
    column = "A" if state[0] == 0 else "B"
    row = str(state[1])
    occupied = "occupied" if state[2] == 1 else "unoccupied"
    parked = "parked" if state[3] == 1 else "unparked"
    context = "({0}, {1}, {2}, {3})".format(column, row, occupied, parked)
    return context
    
def number_action(num):
    if num == 0:
        return "PARK"
    elif num == 1:
        return "DRIVE"
    else:
        return "EXIT"



mdp = P_MDP(10, -1000, -10000, -1, 10)

with open('P_MDP1.txt', 'w') as f:
    f.write('{0} {1}\n\n'.format(mdp.nS, mdp.nA))
    for a in range(mdp.nA):
        transition = '\n'.join(' '.join('{0:0.8f}'.format(elem) for elem in row) for row in mdp.T[a])
        f.write('{0}\n\n'.format(transition))
    f.write(' '.join('{0:0.8f}'.format(R) for R in mdp.R))
    f.write('\n')
f.close()
    
policy1, V1 = infMDP_policy_iter.policy_iter(mdp, 0.9)

with open('P_MDP1_policy.txt', 'w') as f:
    for s in range(mdp.nS - 1):
        f.write('{0}\t\t{1}\n'.format(counter_context(s, mdp), number_action(policy1[s])))
    f.write('terminal_state\t\tNA\n')
f.close()

with open('P_MDP1_value.txt', 'w') as f:
    for s in range(mdp.nS - 1):
        f.write('{0}\t\t{1:.2f}\n'.format(counter_context(s, mdp), V1[s]))
    f.write('terminal_state\t\t1.00\n')
f.close()

        
    

mdp = P_MDP(10, -1000, -10000, -5, 10)

with open('P_MDP2.txt', 'w') as f:
    f.write('{0} {1}\n\n'.format(mdp.nS, mdp.nA))
    for a in range(mdp.nA):
        transition = '\n'.join(' '.join('{0:0.8f}'.format(elem) for elem in row) for row in mdp.T[a])
        f.write('{0}\n\n'.format(transition))
    f.write(' '.join('{0:0.8f}'.format(R) for R in mdp.R))
    f.write('\n') 
f.close()
    
policy2, V2 = infMDP_policy_iter.policy_iter(mdp, 0.9)

with open('P_MDP2_policy.txt', 'w') as f:
    for s in range(mdp.nS - 1):
        f.write('{0}\t\t{1}\n'.format(counter_context(s, mdp), number_action(policy2[s])))
    f.write('terminal_state\t\tNA\n')
f.close()

with open('P_MDP2_value.txt', 'w') as f:
    for s in range(mdp.nS - 1):
        f.write('{0}\t\t{1:.2f}\n'.format(counter_context(s, mdp), V2[s]))
    f.write('terminal_state\t\t1.00\n')
f.close()

    