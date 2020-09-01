# A Deep Reinforcement Learning Algorithm Playing the game Get10

Get10 is a simple game that is typically played on a 5x5 table with integer cells. If a cell has at least one other cell with same integer value, clicking that cell will erase the value of all neighboring same integer cells and the clicked cell will increase in value by one. The erased cells are replaced by new randomly valued cells. Initially, table has integer values between 1 and 4. The goal is get 10 (or higher).

## get10.py
This file contains the get10 class which implements the gameplay.

## learning.py
A self implemented q-learning algorithm

## tf-drl.py
Tensorflow implementation of the q-learning algorithm.

