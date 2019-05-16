#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:13:06 2019

@author: geoffreycideron
"""

"""training procedure"""

import random
import datetime
import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_minigrid
import torch
import torch.nn.functional as F
import tensorboardX as tb

import model
import replaymemory



# Parse arguments

parser = argparse.ArgumentParser(prog="training script")

parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
#parser.add_argument("--seed", type=int, default=0,
#                    help="random seed (default: 0)")
parser.add_argument("--device", type=str,
                    help="cpu or cuda")
parser.add_argument("--model_dir", 
                    help="path to the model's parameters ")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor (default=: 0.99)")
parser.add_argument("--T-max", type=float, default=250,
                    help="Maximum number of steps in an episode (default: 250)")
parser.add_argument("--run", type=int, default=0,
                    help="Numero of the run (default: 0)")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="learning rate (default: 5e-4)")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size (default: 128)")
parser.add_argument("--episodes", type=int, default=200,
                    help="number of episodes during training (default: 200)")
parser.add_argument("--eps-init", type=float, default=1,
                    help="epsilon (exploration) initial (default: 1)")
parser.add_argument("--eps-final", type=float, default=0.1,
                    help="epsilon (exploration) finnal (default: 0.1) ")
parser.add_argument("--T-exploration", type=float, default=1e4,
                    help="Time to reach the minimum exploration (default: 1e4)")
parser.add_argument("--update-target", type=float, default=1e4,
                    help="update the target network every (default: 1e4)")
parser.add_argument("--frames", type=float, default=4,
                    help="Number of frames used to make a state (default: 4)")
parser.add_argument("--memory-size", type=float, default=1e5,
                    help="Size of the replay buffer (default: 1e5)")
parser.add_argument("--summary-q", type=str, default="50,100,150,200",
                    help="log summaries of the distribution of actions (default:50,100,150,200)")
parser.add_argument("--save-model", type=float, default=100,
                    help="Save model parameters every (in episodes) (default: 100)")
parser.add_argument("--ratio", type=float, default=1,
                    help="ratio between the number of transitions and optimization")


args = parser.parse_args()

if not args.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device
    
# Directory to save the model
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
model_dir = args.model_dir or os.path.join(os.getcwd(), "dqn"+"-"+suffix)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    
# Path to save models' parameters 
path_to_save = os.path.join(model_dir, "run_{}/model_params".format(args.run))

if not os.path.exists(path_to_save):
    os.mkdir(os.path.join(model_dir, "run_{}/".format(args.run)))
    os.mkdir(os.path.join(model_dir, "run_{}/model_params/".format(args.run)))

# Summaries (add run{i} for each run)
writer = tb.SummaryWriter(log_dir=model_dir+"/run_{}/logs".format(args.run))

# Create the enviromnent 
env = gym.make(args.env)
observation = env.reset()
# height, width, number of channels
(h,w,c) = observation['image'].shape
# Number and name of actions
n_actions = env.action_space.n
action_names = [a.name for a in env.actions]

# transform the string into a list
summary_q = [int(item) for item in args.summary_q.split(",")]

#%%

steps_done = 0


# Networks
policy_dqn = model.DQN(h, w, c, n_actions, args.frames)
target_dqn = model.DQN(h,w, c, n_actions, args.frames)
target_dqn.load_state_dict(policy_dqn.state_dict())
target_dqn.eval()

# Optimizer
optimizer = torch.optim.RMSprop(policy_dqn.parameters(), lr=args.lr)

# Replay memory
memory = replaymemory.ReplayMemory(size=args.memory_size)

# Selection of an action
def select_action(curr_state):
    global steps_done
    epsilon = max(args.eps_init - steps_done * (args.eps_init - args.eps_final) / args.T_exploration, args.eps_final)
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = policy_dqn(curr_state).max(1)[1].detach() # max(1) for the dim, [1] for the indice, [0] for the value
    steps_done +=1
    # Summaries
    writer.add_scalar("epsilon", epsilon, global_step=steps_done)
    return action

# Optimize the model
def optimize_model():
    if len(memory) < args.batch_size:
        return
    # Sample from the memory replay
    transitions = memory.sample(args.batch_size)
    # Batch the transitions into one namedtuple
    batch_transitions = memory.transition(*zip(*transitions))
    batch_curr_state = torch.cat(batch_transitions.curr_state)
    batch_next_state = torch.cat(batch_transitions.next_state)
    batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype = torch.int32)
    batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long).reshape(-1,1)
    # Compute targets according to the Bellman eq
    targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32)
    targets[batch_terminal==0] += args.gamma * target_dqn(batch_next_state[batch_terminal==0]).max(1)[0].detach()
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
    
    


def training(nb_episodes = args.episodes, T_max = args.T_max, gamma = args.gamma):
    for ep in range(nb_episodes):
        # New episode
        observation = env.reset()
        state = [observation['image']]
        reward_ep = 0
        discounted_reward_ep = 0
        # First frames to make a state
        for _ in range(args.frames-1):
            action = env.action_space.sample()
            observation, reward, terminal, info = env.step(action)
            state.append(observation['image'])
        state = torch.as_tensor(np.concatenate(state, axis=2).transpose(), dtype = torch.float32)[None,:]
        for t in range(T_max):
            # Update the current state
            curr_state = state
            # Select an action
            action = select_action(curr_state)
            # Interaction with the environment
            observation, reward, terminal, info = env.step(action)
            observation_prepro = torch.as_tensor(observation['image'].transpose(), dtype = torch.float32)[None,:]
            state = torch.cat((curr_state[:,c:], observation_prepro), dim=1)

            #print("egualité des 3 dernières frames", torch.equal(curr_state[:,c:], state[:,:9]))
            #print("eq entre les deux states", torch.equal(curr_state, state))
            # Add transition
            memory.add_transition(curr_state, action, reward, state, terminal)
            # Optimization
            optimize_model()
            # Summaries
            writer.add_scalar("length memory", len(memory), global_step=steps_done)
            # Cumulated reward: attention the env gives a reward = 1- 0.9* step_count/max_steps
            reward_ep += reward
            discounted_reward_ep += gamma**t * reward
            # Display the distribution of Q for a state
            if ep+1 in summary_q:
                if t < 50:
                    image = env.render("rgb_array")
                    fig = plt.figure(figsize=(10, 6))
                    ax1 = fig.add_subplot(1,2,2)
                    ax1.set_title("Actions")
                    ax1.bar(range(n_actions), policy_dqn(curr_state).data.numpy().reshape(-1))
                    ax1.set_xticks(range(n_actions))
                    ax1.set_xticklabels(action_names, fontdict=None, minor=False)
                    ax1.set_ylabel("Q values")
                    ax2 = fig.add_subplot(1,2,1)
                    ax2.set_title("Observations")
                    ax2.imshow(image)
                    writer.add_figure("Q values episode {}".format(ep+1), fig, global_step=t)

#            if steps_done % args.summary_q == 0:
#                image = env.render("rgb_array")
#                fig = plt.figure(figsize=(10, 6))
#                ax1 = fig.add_subplot(1,2,2)
#                ax1.set_title("Actions")
#                ax1.bar(range(n_actions), policy_dqn(curr_state).data.numpy().reshape(-1))
#                ax1.set_xticks(range(n_actions))
#                ax1.set_xticklabels(action_names, fontdict=None, minor=False)
#                ax2 = fig.add_subplot(1,2,1)
#                ax2.set_title("Observations")
#                ax2.imshow(image)
#                writer.add_figure("Q(s,.)", fig, global_step=steps_done)
#            # Terminate the episode if terminal state
            if terminal:
                break
        # Update the target network
        if ep % args.update_target == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            writer.add_scalar("time target updated", steps_done, global_step = steps_done)
        # Save policy_dqn's parameters
        if (ep+1) % args.save_model == 0:
            curr_path_to_save = os.path.join(path_to_save, "model_ep_{}.pt".format(ep+1))
            torch.save(policy_dqn.state_dict(),curr_path_to_save)
        # Summaries: cumulated reward per episode & length of an episode
        writer.add_scalar("cum reward per ep", reward_ep, global_step=ep)
        writer.add_scalar("cum discounted reward per ep", discounted_reward_ep, global_step=ep)
        writer.add_scalar("length episode", t, global_step=ep)
    env.close()
    writer.close()
        
#%%
t = time.time()
training()
print("FPS {}".format(round(steps_done/(time.time()-t),2)))
