import os

import numpy as np
import gym_minigrid.envs.game as game
import torch
import torch.nn.functional as F
import tensorboardX as tb
import dill

import models
import replay_memory
import utils as utils
from evaluate import evaluate

"""training procedure"""


def training(dict_env, dict_agent, dict_expert):

    # Type of expert to use
    use_her = 0
    use_learned_expert = 0
    use_noisy_her = 0
    use_dense = 0
    use_expert_to_learn = 0
    start_use_expert = 0
    start_use_dense_expert = 0

    if "her" in dict_expert["name"]:
        use_her = 1
    elif "learned-expert" in dict_expert["name"]:
        use_learned_expert = 1
        start_use_expert = 1
    elif "learned-dense-expert" in dict_expert["name"]:
        use_learned_expert = 1
        use_dense = 1
        start_use_expert = 1
        start_use_dense_expert = 1
    elif "expert-to-learn" in dict_expert["name"]:
        use_expert_to_learn = 1
    elif "expert-dense-to-learn" in dict_expert["name"]:
        use_expert_to_learn = 1
        use_dense = 1
    elif "noisy-her" in dict_expert["name"]:
        use_noisy_her = 1
        use_her = 1
    elif "no-expert" in dict_expert["name"]:
        pass
    else:
        raise Exception("Expert {} not implemented".format(dict_expert["name"]))

    # Device to use
    if "device" in dict_env:
        device = dict_env["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory to save the model
    path_save_model = dict_agent["agent_dir"] + "/model_params"
    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)
    # Summaries
    writer = tb.SummaryWriter(dict_agent["agent_dir"])  # + "/logs")

    # Create the environment
    env = game.game(dict_env)

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
    frames = dict_agent["frames"]
    # The network that predicts the mission only use the last frame
    keep_frames = frames * (c-1)
    # Number and name of actions
    n_actions = env.action_space.n
    action_names = [a.name for a in env.actions]

    # Number of types + colors
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" and "SHADE_TO_IDX" and "SIZE_TO_IDX" in dict_env.keys():
        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())
        num_shade = len(dict_env["SHADE_TO_IDX"].keys())
        num_size = len(dict_env["SIZE_TO_IDX"].keys())
        dim_tokenizer = num_colors + num_types + num_shade + num_size
    else:
        # The mission is not used
        dim_tokenizer = 1

    # F1 score of the dense network

    # Load the model for the learned expert
    if use_learned_expert or use_expert_to_learn:
        if dict_expert["expert_type"] == "onehot":
            net_expert = models.PredMissionOneHot(c=c, frames=1, n_type=num_types, n_color=num_colors,
                                                  n_shade=num_shade, n_size=num_size, lr=dict_agent["lr"])
        if dict_expert["expert_type"] == "dense":
            net_expert = models.PredMissionOneHotDense(c=c, frames=1, n_type=num_types,
                                                       n_color=num_colors, n_shade=num_shade,
                                                       n_size=num_size, lr=dict_agent["lr"])
            # F1 score of the dense network
            f1 = 0
        if dict_expert["expert_type"] == "rnn":
            idx2word = {}
            for key, ind in dict_env["word2idx"].items():
                idx2word[ind] = key

            net_expert = models.PredMissionRNN(c=c, frames=1, n_words=dict_expert["n_words"],
                                               word_embedding_size=dict_expert["word_embedding_size"],
                                               hidden_size=dict_expert["hidden_size"],
                                               teacher_forcing_ratio=dict_expert["teacher_forcing_ratio"],
                                               word2idx=dict_env["word2idx"],
                                               idx2word=idx2word,
                                               lr=dict_expert["lr"],
                                               weight_decay=dict_expert["weight_decay"])
        net_expert.to(device)

    net_expert.load_state_dict(torch.load(dict_expert["expert_weights_path"]))
    net_expert.eval()

    # Define the agent
    if "per" in dict_agent["name"]:
        use_per = 1
    else:
        use_per = 0

    if "dueling" in dict_agent["name"]:
        if use_per:
            agent = models.DuelingDoubleDQNPER
        else:
            agent = models.DuelingDoubleDQN
    else:
        if use_per:
            agent = models.DoubleDQNPER
        else:
            agent = models.DoubleDQN

    params = (h, w, c, n_actions, frames,
              dict_agent["lr"], dim_tokenizer, device, dict_agent["use_memory"], dict_agent["use_text"])

    policy_net = agent(*params).to(device)
    target_net = agent(*params).to(device)

    policy_net.load_state_dict(torch.load(dict_agent["weights_path"]))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Replay memory
    if use_per:
        memory = replay_memory.PrioritizedReplayMemory(size=dict_agent["memory_size"],
                                                      seed=seed, alpha=dict_agent["alpha"], beta=dict_agent["beta"],
                                                      annealing_rate=dict_agent["annealing_rate"])
    else:
        memory = replay_memory.ReplayMemory(size=dict_agent["memory_size"], seed=seed)

    # Warm start the replay buffer
    if dict_agent["warm_start_replay_buffer"]:
        with open(dict_agent["replay_buffer_path"], 'rb') as file:
            memory = dill.load(file)

    # Max steps per episode
    T_MAX = min(dict_env["T_max"], env.max_steps)

    # Number of times the agent interacted with the environment
    steps_done = 0
    episode = 0

    # Starting of the training procedure
    max_steps_reached = 0

    # Evaluate before continual learning
    evaluate(dict_env=dict_env, dict_agent=dict_agent, policy_net=policy_net,
             net_expert=net_expert, writer=writer, global_step=0)

    while True:
        # New episode
        episode += 1
        state = {}
        observation = env.reset()
        # Erase stored transitions (used for every experts)
        memory.erase_stored_transitions()

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
        state_frames = [observation["image"]] * frames
        state_frames = np.concatenate(state_frames, axis=2).transpose((2, 0, 1))
        state_frames = torch.as_tensor(state_frames, dtype=torch.float32).unsqueeze(0)
        state["image"] = state_frames.to(device)

        for t in range(T_MAX):
            # Update the current state
            curr_state = state.copy()

            # Select a random action
            action = np.random.randint(n_actions)

            # Interaction with the environment
            # step_continual: even if an object is picked, the episode continues
            #out_step = env.step_continual(action)
            #observation, reward, terminal, can_pickup = out_step[0], out_step[1], out_step[2], out_step[3]
            out_step = env.step(action)
            observation, reward, terminal, is_carrying = out_step[0], out_step[1], out_step[2], out_step[3]

            # Timeout
            if t == T_MAX - 1:
                terminal = True

            # Update of the next state
            observation_prep \
                = torch.as_tensor(observation["image"].transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)

            state_frames = torch.cat((curr_state["image"][:, c:], observation_prep), dim=1)
            state["image"] = state_frames

            # Update the number of steps
            steps_done += 1

            # Useful for her, and learned expert
            if use_her or use_expert_to_learn or use_learned_expert:
                # The reward by the environment is not used
                memory.store_transition(curr_state=curr_state["image"], action=action, reward=0,
                                    next_state=state["image"], terminal=terminal, mission=curr_state["mission"])

            if start_use_dense_expert and start_use_expert:
                if net_expert.prediction_dense(curr_state["image"][:, keep_frames:]) == 1 and reward < 1:
                    with torch.no_grad():
                        pred_mission = net_expert.prediction_mission(curr_state["image"][:, keep_frames:]).to(device)
                    # Note: terminal is changed to true
                    memory.add_dense_transitions(reward=1, mission=pred_mission, action=3,
                                                 keep_last_transitions_dense=dict_expert["keep_last_transitions_dense"])

            # Optimization
            if steps_done % dict_agent["ratio_step_optim"] == 0:
                policy_net.optimize_model(memory, target_net, dict_agent)

            # Update the target network
            if steps_done % dict_agent["update_target"] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Save policy_net's parameters
            #if steps_done % dict_env["save_model"] == 0:
            #    curr_path_to_save = os.path.join(path_save_model, "model_steps_{}.pt".format(steps_done))
            #    torch.save(policy_net.state_dict(), curr_path_to_save)

            if steps_done % dict_agent["evaluate_policy"] == 0:
                evaluate(dict_env=dict_env, dict_agent=dict_agent, policy_net=policy_net,
                                  net_expert=net_expert, writer=writer, global_step=steps_done)

            if steps_done > dict_env["max_steps"] - 1:
                max_steps_reached = 1
                break

            #if can_pickup:
            #    expert_reward = dict_agent["gamma"] * 1
            #    with torch.no_grad():
            #        expert_mission = net_expert.prediction_mission(curr_state["image"][:, keep_frames:]).to(
            #            device)
            #    memory.add_hindsight_transitions(reward=expert_reward, mission=expert_mission,
            #                                     keep_last_transitions=dict_expert["keep_last_transitions"])
            #    memory.erase_stored_transitions()

            if is_carrying:
                expert_reward = 1
                with torch.no_grad():
                    expert_mission = net_expert.prediction_mission(curr_state["image"][:, keep_frames:]).to(
                        device)
                memory.add_hindsight_transitions(reward=expert_reward, mission=expert_mission,
                                                 keep_last_transitions=dict_expert["keep_last_transitions"])
                memory.erase_stored_transitions()

            if terminal:
                break

        if max_steps_reached:
            break

    env.close()
    writer.close()


# To debug :

with open('configs/envs/fetch_hold_out_missions_50_percent.json', 'r') as myfile:
    config_env = myfile.read()

with open('configs/agents/fetch/duelingdoubledqn_continual.json', 'r') as myfile:
    config_agent = myfile.read()

with open('configs/experts/learned_expert_continual.json', 'r') as myfile:
    config_expert = myfile.read()
import json

dict_env = json.loads(config_env)
dict_agent = json.loads(config_agent)
dict_agent["agent_dir"] = dict_env["env_dir"] + "/" + dict_env["name"] + "/" + dict_agent["name"]
dict_expert = json.loads(config_expert)
print("Training in progress")
training(dict_env, dict_agent, dict_expert)