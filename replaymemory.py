#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:55:52 2019

@author: geoffreycideron
"""

import collections
import random


class ReplayMemory():
    def __init__(self, size=1e6):
        self.transition = collections.namedtuple("Transition", ["curr_state", "action", "reward", "next_state", "terminal"])
        self.memory = []
        self.memory_size = size
        
    
    def add_transition(self, curr_state, action, reward, next_state, terminal):
        self.memory.append(self.transition(curr_state, action, reward, next_state, terminal))
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[1:]
        
    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)