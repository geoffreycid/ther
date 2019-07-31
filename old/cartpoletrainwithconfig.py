import random
import os

import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_minigrid
import torch
import torch.nn.functional as F
import tensorboardX as tb

import model
import replaymemory


"""training procedure"""

# Global variable that count the number of steps



def training(dict_env, dict_agent):
    """

    :type dict_agent: dict of the agent
    ;type dict_env: dict of the environment
    """
    # Device to use
    if "device" in dict_env:
        device = dict_env["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps_done = 0

    # Directory to save the model
    path_save_model = dict_agent["agent_dir"] + "/model_params"
    if not os.path.exists(path_save_model):
        os.mkdir(path_save_model)

    # Summaries (add run{i} for each run)
    writer = tb.SummaryWriter(log_dir=dict_agent["agent_dir"] + "/logs")

    # Create the environment
    env = gym.make(dict_env["env"])
    observation = env.reset()
    # height, width, number of channels
    #(h, w, c) = observation['image'].shape
    # Number and name of actions
    n_actions = env.action_space.n
    #action_names = [a.name for a in env.actions]

    # Networks
    policy_net = model.DQN_cartpole(n_actions, dict_agent["frames"])
    target_net = model.DQN_cartpole(n_actions, dict_agent["frames"])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=dict_agent["lr"])

    # Replay memory
    memory = replaymemory.ReplayMemory(size=dict_agent["memory_size"])

    # Selection of an action
#    def select_action(curr_state):
#        global steps_done
#        epsilon = max(dict_agent["eps_init"] -  steps_done * (dict_agent["eps_init"] - dict_agent["eps_final"])
#                      / dict_agent["T_exploration"], dict_agent["eps_final"])
#        if random.random() < epsilon:
#            action = env.action_space.sample()
#        else:
#            # max(1) for the dim, [1] for the indice, [0] for the value
#            action = int(policy_dqn(curr_state).max(1)[1].detach())
#        # Update the number of steps done
#        steps_done += 1
#        # Summaries
#        writer.add_scalar("epsilon", epsilon, global_step=steps_done)
#        return action

    # Optimize the model
    def optimize_model():
        if len(memory) < dict_agent["batch_size"]:
            return
        optimizer.zero_grad()
        # Sample from the memory replay
        transitions = memory.sample(dict_agent["batch_size"])
        # Batch the transitions into one namedtuple
        batch_transitions = memory.transition(*zip(*transitions))
        batch_curr_state = torch.cat(batch_transitions.curr_state)
        batch_next_state = torch.cat(batch_transitions.next_state)
        batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype=torch.int32)
        batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long).reshape(-1, 1)
        # Compute targets according to the Bellman eq
        targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32)
        targets[batch_terminal == 0] = targets[batch_terminal == 0] \
                                       + dict_agent["gamma"] \
                                       * target_net(batch_next_state[batch_terminal == 0]).max(1)[0].detach()
        targets = targets.reshape(-1, 1)
        # Compute the current estimate of Q
        predictions = policy_net(batch_curr_state).gather(1, batch_action)
        # Loss
        loss = F.smooth_l1_loss(predictions, targets)
        # Optimization

        loss.backward()
        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # Do the gradient descent step
        optimizer.step()

    # Starting of the training procedure
    for episode in range(dict_env["episodes"]):
        # New episode
        observation = env.reset()
        #frames_list = [observation['image']]
        reward_ep = 0
        discounted_reward_ep = 0
        # First frames to make a state
        #for _ in range(dict_agent["frames"] - 1):
        #    action = env.action_space.sample()
        #    observation, reward, terminal, info = env.step(action)
        #    frames_list.append(observation['image'])
        #state = torch.as_tensor(np.concatenate(frames_list, axis=2).transpose(), dtype=torch.float32)[None, :]
        state = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        for t in range(dict_env["T_max"]):
            # Update the current state
            curr_state = state
            # Update epsilon
            epsilon = max(dict_agent["eps_init"] - steps_done * (dict_agent["eps_init"] - dict_agent["eps_final"])
                          / dict_agent["T_exploration"], dict_agent["eps_final"])
            # Select an action
            action = policy_net.select_action(curr_state, epsilon)
            # Interaction with the environment
            observation, reward, terminal, info = env.step(action)
            #observation_prep = torch.as_tensor(observation['image'].transpose(), dtype=torch.float32)[None, :]
            #state = torch.cat((curr_state[:, c:], observation_prep), dim=1)
            state = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            # Update the number of steps
            steps_done += 1
            # Summaries
            writer.add_scalar("epsilon", epsilon, global_step=steps_done)
            # Add transition
            if curr_state.shape != torch.Size([1, 4]):
                print("not same size")
            memory.add_transition(curr_state, action, reward, state, terminal)
            # Optimization
            optimize_model()
            # Summaries
            writer.add_scalar("length memory", len(memory), global_step=steps_done)
            # Cumulated reward: attention the env gives a reward = 1- 0.9* step_count/max_steps
            reward_ep += reward
            discounted_reward_ep += dict_agent["gamma"] ** t * reward
            # Display the distribution of Q for 50 states
            #if episode + 1 in dict_env["summary_q"]:
            #    if t < 50:
            #        image = env.render("rgb_array")
            #        fig = plt.figure(figsize=(10, 6))
            #        ax1 = fig.add_subplot(1, 2, 2)
            #        ax1.set_title("Actions")
            #        ax1.bar(range(n_actions), policy_net(curr_state).data.numpy().reshape(-1))
            #        ax1.set_xticks(range(n_actions))
            #        ax1.set_xticklabels(action_names, fontdict=None, minor=False)
            #        ax1.set_ylabel("Q values")
            #        ax2 = fig.add_subplot(1, 2, 1)
            #        ax2.set_title("Observations")
            #        ax2.imshow(image)
            #        writer.add_figure("Q values episode {}".format(episode + 1), fig, global_step=t)
            if "summary_max_q" in dict_env.keys():
                if steps_done % dict_env["summary_max_q"] == 0 and steps_done > dict_agent["batch_size"]:
                    # Sample a batch of states and compute mean and max values of Q
                    transitions = memory.sample(dict_agent["batch_size"])
                    batch_transitions = memory.transition(*zip(*transitions))
                    batch_curr_state = torch.cat(batch_transitions.curr_state)
                    Q_values = policy_net(batch_curr_state).detach()
                    writer.add_scalar("mean Q", torch.mean(Q_values).data.numpy().reshape(-1)
                                      , global_step=steps_done)
                    writer.add_scalar("max Q", torch.max(Q_values).data.numpy().reshape(-1)
                                      , global_step=steps_done)

            # Terminate the episode if terminal state
            if terminal:
                break
        # Update the target network
        if (episode+1) % dict_agent["update_target"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
            writer.add_scalar("time target updated", episode+1, global_step=steps_done)
        # Save policy_net's parameters
        if (episode + 1) % dict_env["save_model"] == 0:
            curr_path_to_save = os.path.join(path_save_model, "model_ep_{}.pt".format(episode + 1))
            torch.save(policy_net.state_dict(), curr_path_to_save)
        # Summaries: cumulated reward per episode & length of an episode
        writer.add_scalar("cum reward per ep", reward_ep, global_step=episode)
        writer.add_scalar("cum discounted reward per ep", discounted_reward_ep, global_step=episode)
        writer.add_scalar("length episode", t, global_step=episode)
    env.close()
    writer.close()

#with open('config_env.json', 'r') as myfile:
#    config_env = myfile.read()

#with open('config_dqn.json', 'r') as myfile:
#    config_agent = myfile.read()

#import json

#dict_env = json.loads(config_env)
#dict_agent = json.loads(config_agent)
#dict_agent["agent_dir"] = dict_env["env_dir"] + "/" + dict_env["name"] + "/" + dict_agent["name"]
#training(dict_env, dict_agent)

