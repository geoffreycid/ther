import os

import numpy as np
import gym_minigrid.envs.game as game
import torch
import tensorboardX as tb

import models
import replaymemory
import summaryutils as utils

"""training procedure"""


def training(dict_env, dict_agent):
    """
    :type dict_agent: dict of the agent
    ;type dict_env: dict of the environment
    """
    #dict_env = config["dict_env"]
    #dict_agent = config["dict_agent"]
    # Device to use
    if "device" in dict_env:
        device = dict_env["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory to save the model
    path_save_model = dict_agent["agent_dir"] + "/model_params"
    if not os.path.exists(path_save_model):
        os.mkdir(path_save_model)

    # Summaries (add run{i} for each run)
    writer = tb.SummaryWriter(dict_agent["agent_dir"]) # + "/logs")

    # Create the environment
    env = game.game(dict_env, dict_agent)

    # Fix all seeds
    seed = dict_agent["seed"]
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    observation = env.reset()
    # height, width, number of channels
    (h, w, c) = observation['image'].shape
    # Number and name of actions
    n_actions = env.action_space.n
    action_names = [a.name for a in env.actions]

    # Number of types + colors
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():
        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())
        dim_tokenizer = num_colors + num_types
    else:
        # The mission is not used
        dim_tokenizer = 1

    # By default do not use her
    use_her = 0

    # Define the agent
    if dict_agent["agent"] == "dqn-vanille":
        agent = models.DQNVanille
        params = (h, w, c, n_actions, dict_agent["frames"], device)
    if dict_agent["agent"] == "dqn":
        agent = models.DQN
        params = (h, w, c, n_actions, dict_agent["frames"], dim_tokenizer, device)
    if dict_agent["agent"] == "double-dqn":
        agent = models.DoubleDQN
        params = (h, w, c, n_actions, dict_agent["frames"], dim_tokenizer, device)
    if dict_agent["agent"] == "dqn-per":
        agent = models.DQNPER
        params = (h, w, c, n_actions, dict_agent["frames"], dim_tokenizer, device)
    if dict_agent["agent"] == "double-dqn-per":
        agent = models.DoubleDQNPER
        params = (h, w, c, n_actions, dict_agent["frames"], dim_tokenizer, device)
    if dict_agent["agent"] == "double-dqn-her":
        agent = models.DoubleDQNHER
        params = (h, w, c, n_actions, dict_agent["frames"], dim_tokenizer, device)
        use_her = 1

    policy_net = agent(*params).to(device)
    target_net = agent(*params).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=dict_agent["lr"])

    # Replay memory
    if dict_agent["agent"] == "dqn-per" or dict_agent["agent"] == "double-dqn-per":
        memory = replaymemory.PrioritizedReplayMemory(size=dict_agent["memory_size"],
                                                      seed=seed, alpha=dict_agent["alpha"], beta=dict_agent["beta"],
                                                      annealing_rate=dict_agent["annealing_rate"])
    else:
        memory = replaymemory.ReplayMemory(size=dict_agent["memory_size"], seed=seed)

    # Max steps per episode
    T_MAX = min(dict_env["T_max"], env.max_steps)

    # Number of times the agent interacted with the environment
    steps_done = 0
    episode = 0

    # Reward per dict_env["smoothing"] steps
    reward_smoothing = 0
    discounted_reward_smoothing = 0
    episode_done_smoothing = 0
    length_episode_done_smoothing = 0

    # Starting of the training procedure
    while steps_done < dict_agent["max_steps"]:
        # New episode
        episode += 1
        state = {}
        observation = env.reset()
        # Reward per episode
        reward_ep = 0
        # Erase stored transitions (used for HER)
        if use_her:
            memory.erase_stored_transitions()
        
        # One hot encoding of the type and the color of the target object
        target = {
            "color": env.targetColor,
            "type": env.targetType
        }
        state["mission"] = utils.mission_tokenizer(dict_env, target, dim_tokenizer).to(device)

        # Stacking frames to make a state
        state_frames = [observation["image"]] * dict_agent["frames"]
        state_frames = torch.as_tensor(np.concatenate(state_frames, axis=2).transpose(), dtype=torch.float32)[None, :]
        state["image"] = state_frames.to(device)
        
        for t in range(T_MAX):
            # Update the current state
            curr_state = state.copy()
            # Update epsilon
            epsilon = max(dict_agent["eps_init"] - steps_done * (dict_agent["eps_init"] - dict_agent["eps_final"])
                          / dict_agent["T_exploration"], dict_agent["eps_final"])
            # Select an action
            action = policy_net.select_action(curr_state, epsilon)
            # Interaction with the environment
            out_step = env.step(action)
            observation, reward, terminal, return_her = out_step[0], out_step[1], out_step[2], out_step[3]
            observation_prep = \
                torch.as_tensor(observation['image'].transpose(), dtype=torch.float32, device=device)[None, :]
            state_frames = torch.cat((curr_state["image"][:, c:], observation_prep), dim=1)
            state["image"] = state_frames
            # Update the number of steps
            steps_done += 1

            # Add transition
            memory.add_transition(curr_state["image"], action, reward, state["image"], terminal, curr_state["mission"])
            if use_her:
                memory.store_transition(curr_state["image"], action, reward,
                                        state["image"], terminal, curr_state["mission"])

            # Optimization
            if steps_done % dict_agent["ratio_step_optim"] == 0:
                policy_net.optimize_model(memory, target_net, optimizer, dict_agent)

            # Update the target network
            if (steps_done + 1) % dict_agent["update_target"] == 0:
                target_net.load_state_dict(policy_net.state_dict())
                #writer.add_scalar("time target updated", steps_done + 1, global_step=episode)

            # Cumulative reward: attention the env gives a reward = 1- 0.9 * step_count/max_steps
            reward_ep += reward
            reward_smoothing += reward
            discounted_reward_smoothing += dict_agent["gamma"] ** t * reward

            # Summaries
            writer.add_scalar("length memory", len(memory), global_step=steps_done)
            writer.add_scalar("epsilon", epsilon, global_step=steps_done)

            # Record a trajectory
            if episode in dict_env["summary_trajectory"]:
                utils.summary_trajectory(env, writer, n_actions, action_names,
                                         policy_net, state, observation, t, episode)
            # Summaries of Q values
            if "summary_mean_max_q" in dict_env.keys():
                if steps_done % dict_env["summary_mean_max_q"] == 0 and steps_done > dict_agent["batch_size"]:
                    utils.summary_mean_max_q(dict_agent, memory, policy_net, writer, steps_done)

            # Summaries: cumulative reward per episode & length of an episode
            if steps_done % dict_env["smoothing"] == 0:
                reward_smoothing /= dict_env["smoothing"]
                discounted_reward_smoothing /= dict_env["smoothing"]
                writer.add_scalar("Mean reward in {} steps".format(dict_env["smoothing"]),
                                  reward_smoothing, global_step=steps_done)
                writer.add_scalar("Mean discounted reward in {} steps".format(dict_env["smoothing"]),
                                  discounted_reward_smoothing, global_step=steps_done)
                writer.add_scalar("Mean length episode during {} steps".format(dict_env["smoothing"]),
                                  length_episode_done_smoothing / episode_done_smoothing, global_step=steps_done)
                # Re-init to 0
                reward_smoothing = 0
                discounted_reward_smoothing = 0
                episode_done_smoothing = 0
                length_episode_done_smoothing = 0

            # Terminate the episode if terminal state
            if terminal:
                if return_her:
                    hindsight_reward = out_step[4]
                    hindsight_target = out_step[5]
                    mission = utils.mission_tokenizer(dict_env, hindsight_target, dim_tokenizer).to(device)
                    memory.add_hindsight_transitions(reward=hindsight_reward, mission=mission)
                break

        # Length and reward of an episode
        #writer.add_scalar("length episode", t, global_step=episode)
        #writer.add_scalar("Reward per episode", reward_ep, global_step=episode)
        episode_done_smoothing += 1
        length_episode_done_smoothing += t


        # Save policy_net's parameters
        if episode % dict_env["save_model"] == 0:
            curr_path_to_save = os.path.join(path_save_model, "model_ep_{}.pt".format(episode))
            torch.save(policy_net.state_dict(), curr_path_to_save)

    env.close()
    writer.close()


# To debug :

#with open('configenvfetch.json', 'r') as myfile:
#    config_env = myfile.read()

#with open('config_dqn.json', 'r') as myfile:
#    config_agent = myfile.read()

#import json

#dict_env = json.loads(config_env)
#dict_agent = json.loads(config_agent)
#dict_agent["agent_dir"] = dict_env["env_dir"] + "/" + dict_env["name"] + "/" + dict_agent["name"]
#print("Training in progress")
#training(dict_env, dict_agent)