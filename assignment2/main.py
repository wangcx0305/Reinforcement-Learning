#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:27:51 2017

@author: wangchunxiao
"""

from mdp import MDP
from mdp_opt import MDP_OPT
import numpy as np

#####Problem 1#########
mdp = MDP()
mdp.loadfile('MDP_example.txt')

H = 10
(Value, Policy) = MDP_OPT.finiteH_OPT(mdp, H)

np.savetxt('Value_example.txt', Value[:, 1:10], fmt = '%.2f')
np.savetxt('Policy_example.txt', Policy, fmt = '%d')

#####Problem 2########
mdp = MDP()
mdp.loadfile('MDP_self.txt')

H = 10
(Value, Policy) = MDP_OPT.finiteH_OPT(mdp, H)

np.savetxt('Value_self.txt', Value[:, 1:10], fmt = '%.2f')
np.savetxt('Policy_self.txt', Policy, fmt = '%d')


#####Problem 3#######
   ###test 1###
mdp = MDP()
mdp.loadfile('MDP1.txt')

H = 10
(Value, Policy) = MDP_OPT.finiteH_OPT(mdp, H)

np.savetxt('Value1.txt', Value[:, 1:10], fmt = '%.2f')
np.savetxt('Policy1.txt', Policy, fmt = '%d')


  ###test 2###
mdp = MDP()
mdp.loadfile('MDP2.txt')

H = 10
(Value, Policy) = MDP_OPT.finiteH_OPT(mdp, H)

np.savetxt('Value2.txt', Value[:, 1:10], fmt = '%.2f')
np.savetxt('Policy2.txt', Policy, fmt = '%d')











