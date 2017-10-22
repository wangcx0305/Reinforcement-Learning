#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:39:17 2017

@author: wangchunxiao
"""

import numpy as np
from mdp import MDP


class P_MDP(MDP):
    def __init__(self, nrows, handicap_R = -1000,
        collision_R = -10000, nopark_R = -1, R_coeff = 10):
        self.nrows = nrows
        self.state_counter = {}
        self.counter_state = {}
        self.nS = 2 * nrows * 2 * 2 + 1
        self.nA = 3
        self.terminal_state = self.nS - 1
        
        self.R = np.zeros(self.nS)
        self.R[self.terminal_state] = 1
        state_counter = 0
        for c in range(0, 2):
            for r in range(0, nrows):
                for o in range(0, 2):
                    for p in range(0, 2):
                        self.counter_state[state_counter] = (c, r, o, p)
                        self.state_counter[(c, r, o, p)] = state_counter
                        if p == 0:
                            self.R[state_counter] = nopark_R
                        elif p == 1:
                            if o == 1:
                                self.R[state_counter] = collision_R
                            else:
                                if r == 0:
                                    self.R[state_counter] = handicap_R
                                else:
                                    self.R[state_counter] = (self.nrows - r) * R_coeff

                        state_counter += 1

        self.T = []
        for a in range(self.nA): #0: park, 1: drive, 2: exit
            self.T.append(np.zeros((self.nS, self.nS)))

        for a in range(self.nA):
            for c in range(0, 2):
                for r in range(0, nrows):
                    for o in range(0, 2):
                        for p in range(0, 2):
                            curs = self.get_state_counter(c, r, o, p)
                            if a == 0:
                                if p == 0:
                                    nexts = self.get_state_counter(c, r, o, 1)
                                    self.T[a][curs, nexts] = 1
                                else:
                                    nexts = self.terminal_state
                                    self.T[a][curs, nexts] = 1

                            elif a == 2:
                                if p == 1:
                                    nexts = self.terminal_state
                                    self.T[a][curs, nexts] = 1

                            elif a == 1:
                                if p == 0:
                                    if c == 0:
                                        if r == 0:
                                            nextc = 1
                                            nextr = 0
                                        else:
                                            nextc = c
                                            nextr = r - 1
                                    elif c == 1:
                                        if r == self.nrows-1:
                                            nextc = 0
                                            nextr = r
                                        else:
                                            nextc = c
                                            nextr = r + 1

                                    nexts1 = self.get_state_counter(nextc, nextr, 0, p)
                                    nexts2 = self.get_state_counter(nextc, nextr, 1, p)
                                    if r == 0:
                                        o_prob = 0.01
                                    else:
                                        o_prob = float((nrows - r) / nrows)

                                    self.T[a][curs, nexts1] = 1 - o_prob
                                    self.T[a][curs, nexts2] = o_prob
                                else:
                                    nexts = self.terminal_state
                                    self.T[a][curs, nexts] = 1

    def get_state_counter(self, c, r, o, p):
        return self.state_counter[(c, r, o, p)]

    def get_counter_state(self, id):

        if id in self.state_counter:
            (c, r, o, p) = self.counter_state[id]
            return (c, r, o, p)
        else:
            return (-1, -1, -1, -1) # terminal state        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
