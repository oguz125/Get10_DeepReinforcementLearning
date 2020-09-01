#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:45:22 2020

@author: apolat
"""

import numpy as np
from tensorflow.keras import  layers, models, losses, optimizers
from collections import Counter
from get10 import get10

class DoubleLayer:
    def __init__(self, size):
        self.size = size
        self.g10 = get10(size = self.size)
        self.W1 = np.random.randn(self.size**2,self.size**2)
        self.W2 = np.random.randn(self.size**2,self.size**2)
        self.W3 = np.random.randn(self.size**2,self.size**2)
        self.b1 = np.random.randn(1,self.size**2)
        self.b2 = np.random.randn(1,self.size**2)
        self.b3 = np.random.randn(1,self.size**2)

        
        
    def RefreshTable(self):
        self.g10.table = np.random.randint(self.size-2, size=(self.size, self.size))+1
        
    def RandomBenchmark(self, steps = 20):
        for i in range(steps):
            valid_nodes = [(i,j) for i in range(self.size) for j in range(self.size) if self.g10.validnode(i, j)]
            if len(valid_nodes) == 0:
                break
            act = valid_nodes[np.random.randint(len(valid_nodes))]
            self.g10.action(act[0],act[1])
            self.reward = self.Reward()
            
    def NumberThenRow(self):
        while True:
            valid_nodes = [(i,j) for i in range(self.size) for j in range(self.size) if self.g10.validnode(i, j)]
            if len(valid_nodes) == 0:
                break
            value = min([self.g10.table[q] for q in valid_nodes])
            act_i = max([q[0] for q in valid_nodes if self.g10.table[q]==value])
            act_j = min([q[1] for q in valid_nodes if self.g10.table[q]==value and q[0]==act_i])
            self.Play(act_i, act_j, True)
        self.reward = self.Reward()


    
    def Linear(self, table, weights, bias):
        return (np.matmul(table.reshape([1,self.size**2]),weights) + bias).reshape(5,5)
    
    def LinearBack(self, table, weights, in_grad):
        grad_x = np.matmul(in_grad, weights) #dim 1,n**2 x n**2,n**2
        grad_w = np.matmul(in_grad,np.stack([table.reshape([self.size**2]) for i in range(self.size**2)])).T #dim 1,n**2 x n**2,n**2
        grad_b = in_grad #dim 1,n**2
        return grad_x, grad_w, grad_b  
    
    def Update(self, param, grad, dampening = .001):
        param += grad*dampening
        return param/sum(param)
    
    def ReLU(self, inp):
        return inp*(inp>0)
    
    def ReLUBack(self, inp, in_grad):
        return in_grad*(inp>0)
    
    def Reward(self):
        count = Counter(self.g10.table.reshape(self.size**2))
        return (sum([i*(1+(count[i]/self.size**2)) for i in range(max(self.g10.table.reshape(self.size**2)))])-21)/4.9
        
    
    def Play(self, i, j, change_table = False):
        if change_table:
            self.g10.action(i,j)
            return self.Reward()
        else:
            A = self.g10.table
            self.g10.action(i,j)
            ret = self.Reward()
            self.g10.table = A
            return ret
                     
        

    
    def Forward(self):
        if self.g10.cont() == 0:
            return
        inp = self.g10.table
        self.il1 = self.Linear(inp,self.W1,self.b1)
        self.ol1 = self.ReLU(self.il1)
        self.il2 = self.Linear(self.ol1, self.W2, self.b2)
        self.ol2 = self.ReLU(self.il2)
        self.il3 = self.Linear(self.ol2, self.W3, self.b3)
        valid_nodes = [(i,j) for i in range(self.size) for j in range(self.size) if self.g10.validnode(i, j)]
        if not valid_nodes:
            return 
        max_out = -np.inf
        for k in valid_nodes:
            if self.il3[k[0],k[1]] > max_out:
                max_out = self.il3[k[0],k[1]]
                act = k
        self.reward = self.Play(act[0],act[1], change_table = True)
        return act
        
    def Simulate(self, steps = 20, trace = False):
        if trace:
            action_list = []
        if steps == 'max':
            while self.g10.cont() > 0:
                q = self.Forward()
                if trace:
                    action_list.append(q)
                    
        else:
            for i in range(steps):
                q = self.Forward()
                if trace:
                    action_list.append(q)
        if trace:
            return action_list
        
    
    def Backward(self,grad):
        gx3, gw3, gb3 = self.LinearBack(self.g10.table, self.W3, grad)
        self.W3 = self.Update(self.W3, gw3)
        self.b3 = self.Update(self.b3, gb3)
        gx3 = self.ReLUBack(self.il2.reshape(self.size**2), gx3)
        gx2, gw2, gb2 = self.LinearBack(self.g10.table, self.W2, gx3)
        self.W2 = self.Update(self.W2, gw2)
        self.b2 = self.Update(self.b2, gb2)
        gx2 = self.ReLUBack(self.il1.reshape(self.size**2), gx2)
        _, gw1, gb1 = self.LinearBack(self.g10.table, self.W1, gx2)
        self.W1 = self.Update(self.W1, gw1)
        self.b1 = self.Update(self.b1, gb1)
        
        
        
    def RandomWalk(self, steps = 20, num_try = 10, epoch=100, save_rewards = False):
        l = []
        for n in range(epoch):
            params = [[self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]]
            self.Simulate(steps)
            if save_rewards:
                l.append(self.reward)
            rewards = [self.reward]
            for i in range(num_try):
                grad = 2*(np.random.rand(1,self.size**2)>.5)-1
                self.Backward(grad)
                self.Simulate(steps)
                rewards.append(self.reward)
                params.append([self.W1, self.W2, self.W3, self.b1, self.b2, self.b3])
            [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3] = params[np.array(rewards).argmax()]
            self.RefreshTable()
        if save_rewards:
            return np.array(l)
            

               
                
    def RandomWalk2(self, steps = 20, epoch=100, save_rewards = False):
        l = []
        for n in range(epoch):
            self.Simulate(steps)
            if save_rewards:
                l.append(self.reward)
            self.R = np.array([[self.Play(i,j) for j in range(self.size)] for i in range(self.size)]).reshape(1,self.size**2)
            grad = self.R
            self.Backward(grad)
            self.RefreshTable()
        if save_rewards:
            return np.array(l)
            
        
    def RandomWalk3(self, steps = 20, epoch=100, save_rewards = False):
        l = []
        for n in range(epoch):
            action_list = self.Simulate(steps, trace=True)
            count = Counter(action_list)
            dist = np.array([[count[(i,j)]/sum(count.values()) for j in range(self.size)] for i in range(self.size)]).reshape(1,self.size**2)
            if save_rewards:
                l.append(self.reward)
            grad = dist*self.reward
            self.Backward(grad)
            self.RefreshTable()
        if save_rewards:
            return np.array(l)
        


'''            
dl = DoubleLayer(5) 
RW1 = dl.RandomWalk('max', epoch = 500, save_rewards=True)
dl = DoubleLayer(5) 
RW2 = dl.RandomWalk2('max', epoch = 500, save_rewards=True)
dl = DoubleLayer(5) 
RW3 = dl.RandomWalk3('max', epoch = 500, save_rewards=True)
'''
        
        


'''
steps = 200
epoch = 1000
num_try = 100
rand = []
walk1 = []    
walk2 = []            
for i in range(500):
    dl = DoubleLayer(5) 
    dl.RandomBenchmark(1000)
    rand.append(dl.reward)
    dl = DoubleLayer(5)
    dl.RandomWalk(steps='max',epoch=epoch)
    walk1.append(dl.reward)
    dl = DoubleLayer(5)
    dl.RandomWalk2(steps='max',epoch=epoch)
    walk2.append(dl.reward)
    
''' 
'''
ss=[]
for i in range(1,60):
    steps = 5*i
    t=[]
    for i in range(100):
        dl = DoubleLayer(5)
        dl.RandomWalk2(steps=steps,epoch=epoch)
        t.append(dl.reward)
    ss.append(sum(t)/100)
'''        

'''
for i in range(100):
    dl = DoubleLayer(5) 
    dl.RandomWalk3(1000)
    rand.append(dl.reward)
    
'''   
'''
perfect=np.ones(25)*10
g10=get10()
model = models.Sequential()
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(25, activation='softmax'))
model.compile(loss=losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])


train_history = model.fit(g10.table.reshape(25), perfect, epochs=1)

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']    
'''   
            
            
            
'''
Sample Mean ~ 21
Sample std ~ 4.9
'''
        
        
'''    
    
    def Reward(self, out, table):
        valid_nodes = [(i,j) for i in range(self.size) for j in range(self.size) if self.g10.validnode(i, j)]
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) not in valid_nodes:
                    out[i,j]=-np.inf
        act = np.unravel_index(out.argmax(), table.shape)
        self.g10.action(act[0],act[1])
        return (sum(sum(self.g10.table**3))-1570)/415
'''     
'''
l = []
for i in range(500):
    dl = DoubleLayer(5)
    dl.NumberThenRow()
    l.append(dl.reward)
'''
    
    
        
    