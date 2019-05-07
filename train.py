#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:13:06 2019

@author: geoffreycideron
"""

"""training procedure"""

 
#logger in tensorboard  cb de fois j'ai optimz dqn), phase evaluation, sum rewards/discounted
#  faire en sorte de pouvoir avoir un writer par runs (facilite quand il faut lancer plusieurs runs) 
import random
import gym
import numpy as np
import gym_minigrid
import model
import replaymemory
import torch
import torch.nn.functional as F
import tensorboardX as tb
#%%

# Summaries (add run{i} for each run)
writer = tb.SummaryWriter(log_dir="./summaries")


# Create the enviromnent 
env = gym.make('MiniGrid-Empty-8x8-v0')

observation = env.reset()
(h,w,_) = observation['image'].shape
n_actions = env.action_space.n

# Params (next step use a sparser)
nb_ep = 50
T = 500
BATCH_SIZE = 128
frames = 4
gamma = 0.99

TARGET_UPDATE = 20

# Annealing epsilon
EPS_INIT = 0.95
EPS_FINAL = 0.1
T_EXPLORATION = 1e4
steps_done = 0

# Compute a summary every x steps
SUMMARY_Q = 100

def select_action(curr_state, policy_dqn):
    global steps_done
    epsilon = max(EPS_INIT - steps_done * (EPS_INIT - EPS_FINAL)/T_EXPLORATION, EPS_FINAL)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = policy_dqn(curr_state).max(1)[1].detach() # max(1) for the dim, [1] for the indice, [0] for the value
    steps_done +=1
    # Summaries
    writer.add_scalar("epsilon", epsilon, global_step=steps_done)
    if steps_done % SUMMARY_Q == 0:
        writer.add_histogram("Q(s,.)", policy_dqn(curr_state).detach(), global_step=steps_done)
    return action


def optimize_model(memory, policy_dqn, target_dqn, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    # Sample from the memory replay
    transitions = memory.sample(BATCH_SIZE)
    # Batch the transitions into one namedtuple
    batch_transitions = memory.transition(*zip(*transitions))
    batch_curr_state = torch.cat(batch_transitions.curr_state)
    batch_next_state = torch.cat(batch_transitions.next_state)
    batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype = torch.int32)
    batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long).reshape(-1,1)
    # Compute targets according to the Bellman eq
    targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32)
    targets[batch_terminal==0] += gamma * target_dqn(batch_next_state[batch_terminal==0]).max(1)[0].detach()
    targets = targets.reshape(-1,1)
    # Compute the current estimate of Q
    preds = policy_dqn(batch_curr_state).gather(1, batch_action)
    # Loss
    loss = F.mse_loss(preds, targets)
    # Optimization
    optimizer.zero_grad()
    loss.backward()
    # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
    for param in policy_dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # Summaries to do 
    


def training(nb_episodes = nb_ep, T_max = T, gamma=0.99, memory_size=1e3):
    # Networks
    policy_dqn = model.DQN(h, w, frames, n_actions)
    target_dqn = model.DQN(h,w, frames, n_actions)
    target_dqn.load_state_dict(policy_dqn.state_dict())
    target_dqn.eval()
    # Optimizer
    optimizer = torch.optim.RMSprop(policy_dqn.parameters())
    # Replay memory
    memory = replaymemory.ReplayMemory(size=memory_size)
    # Preprocessing
    for ep in range(nb_episodes):
        # New episode
        observation = env.reset()
        state = [observation['image']]
        reward_ep = 0
        discounted_reward_ep = 0
        # First frames to make a state
        for _ in range(frames-1):
            env.render()
            action = env.action_space.sample()
            observation, reward, terminal, info = env.step(action)
            state.append(observation['image'])
        state = torch.as_tensor(np.concatenate(state, axis=2).transpose(), dtype = torch.float32)[None,:]
        
        for t in range(T_max):
            curr_state = state
            env.render()
            # Select an action
            action = select_action(curr_state, policy_dqn)
            # Interaction with the environment
            observation, reward, terminal, info = env.step(action)
            observation_prepro = torch.as_tensor(observation['image'].transpose(), dtype = torch.float32)[None,:]
            state = torch.cat((curr_state[:,3:],observation_prepro), dim=1)
            # Add transition
            memory.add_transition(curr_state, action, reward, state, terminal)
            # Optimization 
            optimize_model(memory, policy_dqn, target_dqn, optimizer)
            # Summaries
            writer.add_scalar("length memory", len(memory), global_step=steps_done)
            # Cumulated reward
            reward_ep += reward
            discounted_reward_ep += gamma**t * reward
            # Terminate the episode if terminal state
            if terminal:
                break
        # Update the target network
        if ep % TARGET_UPDATE == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            writer.add_scalar("time target updated", steps_done, global_step=steps_done)
        # Summaries: cumulated reward per episode & length of an episode
        writer.add_scalar("cum reward per ep", reward_ep, global_step=ep)
        writer.add_scalar("cum discounted reward per ep", discounted_reward_ep, global_step=ep)
        writer.add_scalar("length episode", t, global_step=ep)
#%%
training()
writer.close()