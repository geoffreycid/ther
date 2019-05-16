#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:54:22 2019

@author: geoffreycideron
"""

"""
Evaluation procedure
"""
import torch
import model
import gym_minigrid
import gym
import argparse
import numpy as np
import collections
import random
import time
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--path_model", required=True, 
                    help="path to the parameters of the model (REQUIRED)")
#parser.add_argument("--model", required=True,
#                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=50,
                    help="number of episodes of evaluation (default: 50)")
#parser.add_argument("--seed", type=int, default=0,
#                    help="random seed (default: 0)")
parser.add_argument("--device", type=str,
                    help="cpu or cuda")
parser.add_argument("--gamma", type=float, default=0.99, 
                   help="discount factor (default=: 0.99)")
parser.add_argument("--T_max", type=float, default=250,
                    help="Maximum number of steps in an episode (default: 250)")
parser.add_argument("--frames", type=float, default=1,
                    help="Frames used to make a state (default: 1)")
args = parser.parse_args()

if not args.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device


# Create the enviromnent 
env = gym.make(args.env)

observation = env.reset()
# height, width, number of channels
(h,w,c) = observation['image'].shape
# Number of actions
n_actions = env.action_space.n
# Number of frames to define a state
frames = args.frames

# Define agent 
net = model.DQN(h, w, c, n_actions, frames=frames).to(device)
net.load_state_dict(torch.load(args.path_model))
net.eval()
# Initialize logs
logs = {"length_episode": [], "return_per_episode": []}

# Synthesize an array with useful statistics
def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d

def select_action(curr_state):
    epsilon = 0.05
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = net(curr_state).max(1)[1].detach() # max(1) for the dim, [1] for the indice, [0] for the value
    return action

# time_begin and steps_done are used to calculate FPS
time_begin = time.time()
steps_done = 0

for _ in range(args.episodes):
    observation = env.reset()
    state = [observation['image']]
    reward_ep = 0
    discounted_reward_ep = 0
    # First frames to make a state
    for _ in range(frames-1):
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        state.append(observation['image'])
    state = torch.as_tensor(np.concatenate(state, axis=2).transpose(), dtype = torch.float32)[None,:]
    for t in range(args.T_max):
        steps_done +=1
        env.render()
        curr_state = state
        # Select an action
        action = select_action(curr_state)
        # Interaction with the environment
        observation, reward, terminal, info = env.step(action)
        observation_prepro = torch.as_tensor(observation['image'].transpose(), dtype = torch.float32)[None,:]
        state = torch.cat((curr_state[:,c:],observation_prepro), dim=1)
        # Cumulated reward: attention the env gives a reward = 1- 0.9* step_count/max_steps
        reward_ep += reward
        discounted_reward_ep += args.gamma**t * reward
        # Terminate the episode if terminal state
        if terminal:
            break
    logs["length_episode"].append(t)
    logs["return_per_episode"].append(reward_ep)
env.close()

lengths_summaries = synthesize(logs["length_episode"])
rewards_summaries = synthesize(logs["return_per_episode"])
print("Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | Length:μσmM {:.1f} {:.1f} {} {} | FPS: {}"
      .format(*rewards_summaries.values(), *lengths_summaries.values(),round(steps_done/(time.time() - time_begin),0)))
