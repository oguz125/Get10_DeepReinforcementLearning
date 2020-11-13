#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:16:09 2020

@author: apolat
"""
#a

import numpy as np


class get10:
    def __init__(self, size = 5):
        self.size = size
        self.table = np.random.randint(self.size - 2, size = (self.size, self.size)) + 1

    # Returns the adjacent cells of (i,j)
    def adjacentnodes(self, i, j):
        ret = []
        if i + 1 in range(self.size):
            ret.append((i + 1, j))
        if i - 1 in range(self.size):
            ret.append((i - 1, j))
        if j + 1 in range(self.size):
            ret.append((i, j + 1))
        if j - 1 in range(self.size):
            ret.append((i, j - 1))
        return ret

    # Returns True if (i,j) has an adjacent node with same value
    def validnode(self, i, j):
        return len([k for k in self.adjacentnodes(i, j) if self.table[i, j] == self.table[k]]) > 0

    # Number of playable nodes
    def cont(self):
        return sum([self.validnode(i, j) for i in range(self.size) for j in range(self.size)])

    # The nodes that is connected to (i,j) with the same value nodes        
    def adjacentregion(self, i, j):
        val = self.table[i, j]
        self.table[i, j] = -1
        for k in self.adjacentnodes(i, j):
            if self.table[k] == val:
                self.adjacentregion(k[0], k[1])

    # Updates the table if (i,j) is played
    def action(self, i, j):
        if self.validnode(i, j):
            new_val = self.table[i, j] + 1
            self.adjacentregion(i, j)
            A = self.table
            A[i, j] = new_val
            for i in range(self.table.shape[0]):
                l = [a for a in A[:, i] if a > 0]
                self.table[:, i] = list(1 + np.random.randint(self.size - 2, size = self.size - len(l))) + l
        return self.table
