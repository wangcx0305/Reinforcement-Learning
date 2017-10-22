#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:54:16 2017

@author: wangchunxiao
"""

import io
import numpy as np


class MDP:
    
    def __init__(self):
        self.nS = 0
        self.nA = 0
        self.T = []
        self.R = []
   
    def loadfile(self, filename):
        fhandle = io.open(filename, 'r')
        file = fhandle.read().splitlines()
        line1 = file[0].replace('\ufeff', '').split()
        ns = int(line1[0])
        na = int(line1[1])
        self.nS = ns
        self.nA = na
        strR = file[-1].split()
        self.R = [float(ele) for ele in strR]
        for j in range(na):
            tran = np.zeros((ns, ns))
            for i in range(ns):
                line = file[2 + (ns + 1) * j + i].split()
                line = [float(ele) for ele in line]
                for k in range(ns):
                   tran[i, k] = line[k]
            self.T.append(tran)
            
    def get_tprob(self, action, curs, nexts = None):
        if nexts is None:
            return self.T[action][curs, :]
        else:
            return self.T[action][curs, nexts]

    def get_reward(self, curs):
        return self.R[curs]
        
        
        
        