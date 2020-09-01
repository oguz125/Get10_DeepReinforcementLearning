#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:42:32 2020

@author: apolat
"""

import numpy as np 
import tensorflow as tf
from collections import Counter

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from tf_agents.environments import tf_py_environment
from tf_agents.networks import network
from tf_agents.environments import tf_environment
from tf_agents.specs import tensor_spec
from tf_agents.policies import q_policy
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.agents.reinforce import reinforce_agent

class get10Env(py_environment.PyEnvironment):
    
    def __init__(self, size = 5):
        self.size = size
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=size**2-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.size**2,), dtype=np.int32, minimum=1, maximum=self.size*4, name='observation')
        self.table = np.random.randint(self.size-2, size=(self.size, self.size))+1
        self._state = self.table.reshape(25)
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
        
    def _reset(self):
        self.table = np.random.randint(self.size-2, size=(self.size, self.size))+1
        self._state = self.table.reshape(25)
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        action_ = (int(action/self.size), action%self.size)
        
        if self._episode_ended:
            return self.reset()
        
        if self.cont()==0:
            self._episode_ended = True
        elif action_[0] in range(self.size) and action_[1] in range(self.size):
            self.Action(action_[0],action_[1])
        else:
            raise ValueError('`action` should be a scalar for the flattened game table.')
            
        if self._episode_ended:
          reward = self.Reward()
          return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
          return ts.transition(
              np.array(self._state, dtype=np.int32), reward=self.Reward(), discount=.5)
        
    def check_valid(self):
        utils.validate_py_environment(self, episodes=5)
        print('OK')
    
    def wrap_env(self):
        return tf_py_environment.TFPyEnvironment(self)
        
    
    def adjacentnodes(self,i,j):
        ret = []
        if i+1 in range(self.size):
            ret.append((i+1,j))
        if i-1 in range(self.size):
            ret.append((i-1,j))
        if j+1 in range(self.size):
            ret.append((i,j+1))
        if j-1 in range(self.size):
            ret.append((i,j-1))
        return ret
        
        
    def validnode(self,i,j):
        return len([k for k in self.adjacentnodes(i,j) if self.table[i,j]==self.table[k]])>0
    
    def cont(self):
        return sum([self.validnode(i,j) for i in range(self.size) for j in range(self.size)])
            
    def adjacentregion(self,i,j):
        val = self.table[i,j]
        self.table[i,j] = -1
        for k in self.adjacentnodes(i,j):
            if self.table[k] == val:
                self.adjacentregion(k[0],k[1])
  
    def Action(self, i,j):
        if self.validnode(i,j):
            new_val = self.table[i,j]+1
            self.adjacentregion(i,j)
            A = self.table
            A[i,j] = new_val
            for i in range(self.table.shape[0]):
                l = [a for a in A[:,i] if a>0]
                self.table[:,i] = list(1+np.random.randint(self.size-2, size=self.size-len(l)))+l
            self._state = self.table.reshape(25)
        return self._state
    
    def Reward(self):
        count = Counter(self._state)
        return sum([i*(1+(count[i]/self.size**2)) for i in range(max(self._state))])
    
    def RandomBenchmark(self, steps = 'max'):
        if steps != 'max':
            i=0
        while True:
            valid_nodes = [(i,j) for i in range(self.size) for j in range(self.size) if self.validnode(i, j)]
            if len(valid_nodes) == 0:
                return self._state
            act = valid_nodes[np.random.randint(len(valid_nodes))]
            self._step(act)
            if steps != 'max':
                i+=1
                if i>steps:
                    return self._state

class QNetwork(network.Network):

  def __init__(self, input_tensor_spec, action_spec, num_actions, name=None):
    super(QNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
    self._sub_layers = [
        tf.keras.layers.Dense(num_actions, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_actions, activation=tf.nn.relu),
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._sub_layers:
      inputs = layer(inputs)
    return inputs, network_state

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)


env = get10Env(5)
tf_env = env.wrap_env()
input_tensor_spec = tf_env.observation_spec()
time_step_spec = ts.time_step_spec(input_tensor_spec)
action_spec = tf_env.action_spec()

num_actions = env.size**2
batch_size = 1
observation = tf.cast([(np.random.randint(env.size-2, size=(env.size, env.size))+1).reshape(25) for _ in range(1)], tf.int32)
time_steps = ts.restart(observation, batch_size=batch_size)

my_q_network = QNetwork(
    input_tensor_spec=input_tensor_spec,
    action_spec=action_spec,
    num_actions=num_actions)
my_q_policy = q_policy.QPolicy(
    time_step_spec, action_spec, q_network=my_q_network)
action_step = my_q_policy.action(time_steps)
distribution_step = my_q_policy.distribution(time_steps)

print('Action:')
print(action_step.action)

print('Action distribution:')
print(distribution_step.action)

num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
observers = [num_episodes, env_steps]
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    tf_env, my_q_policy, observers, num_episodes=2)

# Initial driver.run will reset the environment and initialize the policy.
final_time_step, policy_state = driver.run()

print('final_time_step', final_time_step)
print('Number of Steps: ', env_steps.result().numpy())
print('Number of Episodes: ', num_episodes.result().numpy())



learning_rate = 0.001
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

agent = reinforce_agent.ReinforceAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    actor_network=my_q_network,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
agent.initialize()



replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=1000)


collect_data(tf_env, agent.policy, replay_buffer, steps=100)


num_eval_episodes = 10
num_iterations = 20000
collect_steps_per_iteration = 1 
log_interval = 200
eval_interval = 1000

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
print('ABOUT TO GO')
avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    print('HERE WE GO')

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(tf_env, agent.collect_policy, replay_buffer)
    
        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
      
        step = agent.train_step_counter.numpy()
      
        if step % log_interval == 0:
          print('step = {0}: loss = {1}'.format(step, train_loss))
      
        if step % eval_interval == 0:
          avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes)
          print('step = {0}: Average Return = {1}'.format(step, avg_return))
          returns.append(avg_return)

