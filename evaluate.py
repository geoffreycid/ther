import os

import numpy as np
import gym_minigrid.envs.game as game
import torch
import torch.nn.functional as F
import tensorboardX as tb

import models
import replaymemory
import summaryutils as utils

"""training procedure"""


def evaluate(dict_env, dict_agent, policy_net, net_expert, writer, global_step):

    # Init expert
    use_learned_expert = 1

    # Device to use
    if "device" in dict_env:
        device = dict_env["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment
    env = game.game(dict_env)

    # Fix all seeds
    seed = dict_agent["seed_evaluate"]
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    observation = env.reset()
    # height, width, number of channels
    (h, w, c) = observation['image'].shape
    # the last channel is close/open
    c = c-1
    frames = dict_agent["frames"]
    # The network that predicts the mission only use the last frame
    keep_frames = frames * (c-1)

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
    # Average timeout per episode
    timeout_smoothing = 0
    # Average time an object was picked during an episode
    object_picked_smoothing = 0
    # Monitor the accuracy of the expert
    pred_mission_smoothing = 0
    episode_done_carrying = 0

    # Success rate useful for negative reward
    success_rate_smoothing = 0

    if not dict_env["wrong_object_terminal"]:
        wrong_object_picked = 0

    while steps_done < dict_agent["max_steps_evaluate"]:
        # New episode
        episode += 1
        state = {}
        observation = env.reset()
        # Reward per episode
        reward_ep = 0

        # One hot encoding of the type and the color of the target object
        target = {
            "color": env.targetColor,
            "type": env.targetType,
            "shade": env.targetShade,
            "size": env.targetSize
        }
        if dict_agent["use_text"]:
            state["mission"] = utils.indexes_from_sentences(observation["mission"], dict_env["word2idx"]).to(device)
        else:
            state["mission"] = utils.mission_tokenizer(dict_env, target).to(device)

        # Stacking frames to make a state
        observation["image"] = observation["image"][:, :, :c]
        state_frames = [observation["image"]] * frames
        state_frames = np.concatenate(state_frames, axis=2).transpose((2, 0, 1))
        state_frames = torch.as_tensor(state_frames, dtype=torch.float32).unsqueeze(0)
        state["image"] = state_frames.to(device)

        for t in range(T_MAX):
            # Update the current state
            curr_state = state.copy()

            # Select a random action
            action = policy_net.select_action(curr_state, epsilon=0.05)

            # Interaction with the environment
            out_step = env.step(action)
            observation, reward, terminal, return_her, is_carrying \
                = out_step[0], out_step[1], out_step[2], out_step[3], out_step[4]

            # Timeout
            if t == T_MAX - 1:
                terminal = True

            # Update of the next state
            observation["image"] = observation["image"][:, :, :c]
            observation_prep \
                = torch.as_tensor(observation["image"].transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)

            state_frames = torch.cat((curr_state["image"][:, c:], observation_prep), dim=1)
            state["image"] = state_frames

            # Update the number of steps
            steps_done += 1

            # Cumulative reward
            reward_ep += reward
            reward_smoothing += reward
            discounted_reward_smoothing += dict_agent["gamma"] ** t * reward

            # Summaries when an episode is ended only when one take the good object or timeout
            if not dict_env["wrong_object_terminal"]:
                if is_carrying and reward < 1:
                    wrong_object_picked += 1

            # Summaries: cumulative reward per episode & length of an episode
            if terminal:
                episode_done_smoothing += 1
                length_episode_done_smoothing += t
                timeout_smoothing += 1 - is_carrying
                object_picked_smoothing += is_carrying
                success_rate_smoothing += reward > 0
                # Monitor the accuracy of the expert
                if is_carrying and reward > 0 and use_learned_expert:
                    expert_mission = net_expert.prediction_mission(curr_state["image"][:, keep_frames:]).to(device)
                    pred_mission_smoothing += torch.equal(torch.sort(expert_mission[-4:])[0],
                                                          torch.sort(curr_state["mission"][-4:])[0])
                    episode_done_carrying += 1

            if terminal:
                break

    writer.add_scalar("Mean reward in {} steps".format(dict_agent["max_steps_evaluate"]),
                      reward_smoothing / dict_agent["max_steps_evaluate"], global_step=global_step)
    writer.add_scalar("Mean discounted reward in {} steps".format(dict_agent["max_steps_evaluate"]),
                      discounted_reward_smoothing / dict_agent["max_steps_evaluate"], global_step=global_step)
    writer.add_scalar("Mean length episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                      length_episode_done_smoothing / episode_done_smoothing, global_step=global_step)
    writer.add_scalar("Mean reward per episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                      reward_smoothing / episode_done_smoothing, global_step=global_step)
    writer.add_scalar("Timeout per episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                      timeout_smoothing / episode_done_smoothing, global_step=global_step)
    writer.add_scalar("Object picked per episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                      object_picked_smoothing / episode_done_smoothing, global_step=global_step)
    writer.add_scalar("Success rate per episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                      success_rate_smoothing / episode_done_smoothing, global_step=global_step)

    print("Success rate", success_rate_smoothing / episode_done_smoothing)

    if not dict_env["wrong_object_terminal"]:
        writer.add_scalar("Wrong object picked rate per episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                          wrong_object_picked / episode_done_smoothing, global_step=global_step)

    episode_done_carrying = max(episode_done_carrying, 1)

    if use_learned_expert:
        writer.add_scalar("Mean acc of pred per episode during {} steps".format(dict_agent["max_steps_evaluate"]),
                      pred_mission_smoothing / episode_done_carrying, global_step=global_step)
        print("Mean acc pred", pred_mission_smoothing / episode_done_carrying)

    env.close()