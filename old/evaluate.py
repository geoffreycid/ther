#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:54:22 2019

@author: geoffreycideron
"""
import json

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
import json
import models
import gym_minigrid.envs.game as game


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model
#path_model = "saved_model/model_ep_5000.pt"
path_model = "/home/gcideron/visual_her/out/Fetch-5x5-N2-C1-1mission/" \
             "double-dqn-wolanguage-rewardpos--4-frames-12000ep/model_params/model_ep_5000.pt"
# Create the enviromnent
with open('configs/envs/fetch.json', 'r') as myfile:
    config_env = myfile.read()
dict_env = json.loads(config_env)

env = game.game(dict_env)

observation = env.reset()
# height, width, number of channels
(h,w,c) = observation['image'].shape
# Number of actions
n_actions = env.action_space.n
# Number of frames to define a state
frames = 4

T_max = 200
episodes = 50
gamma = 0.99

# Define agent 
net = models.DoubleDQN(h, w, c, n_actions, frames=frames, dim_tokenizer=1, device=device).to(device)
net.load_state_dict(torch.load(path_model, map_location='cpu'))
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


# time_begin and steps_done are used to calculate FPS
time_begin = time.time()
steps_done = 0

# Starting of the training procedure
for episode in range(episodes):
    # New episode
    state = {}
    observation = env.reset()
    # Reward per episode
    reward_ep = 0
    discounted_reward_ep = 0

    # One hot encoding of the type and the color of the target object
    #if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():
    #    mission_color_onehot = torch.zeros(num_colors)
    #    # mission_color_onehot[dict_env["COLOR_TO_IDX"][env.targetColor]] = 1
    #    mission_type_onehot = torch.zeros(num_types)
    #    mission_type_onehot[dict_env["TYPE_TO_IDX"][env.targetType]] = 1
    #    state["mission"] = torch.cat((mission_color_onehot, mission_type_onehot))[None, :].to(device)
    #else:
        # Constant mission
    state["mission"] = torch.zeros(1, 1, device=device)

    # Stacking frames to make a state
    state_frames = [observation["image"]]
    # First frames to make a state
    for _ in range(frames - 1):
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        state_frames.append(observation['image'])
    state_frames = torch.as_tensor(np.concatenate(state_frames, axis=2).transpose(), dtype=torch.float32)[None, :]
    state["image"] = state_frames.to(device)

    for t in range(T_max):
        env.render()
        # Update the current state
        curr_state = state.copy()

        # Select an action
        action = net.select_action(curr_state, 0.05)

        # Interaction with the environment
        observation, reward, terminal, info = env.step(action)
        observation_prep = \
            torch.as_tensor(observation['image'].transpose(), dtype=torch.float32, device=device)[None, :]
        state_frames = torch.cat((curr_state["image"][:, c:], observation_prep), dim=1)
        state["image"] = state_frames

        # Update the number of steps
        steps_done += 1

        # Cumulated reward: attention the env gives a reward = 1- 0.9* step_count/max_steps
        reward_ep += reward
        discounted_reward_ep += gamma ** t * reward
        # Terminate the episode if terminal state
        if terminal:
            print("reward episode", reward)
            break
    logs["length_episode"].append(t)
    logs["return_per_episode"].append(reward_ep)
env.close()

lengths_summaries = synthesize(logs["length_episode"])
rewards_summaries = synthesize(logs["return_per_episode"])
print("Reward:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | Length:μσmM {:.1f} {:.1f} {} {} | FPS: {}"
      .format(*rewards_summaries.values(), *lengths_summaries.values(),
              round(steps_done / (time.time() - time_begin), 0)))