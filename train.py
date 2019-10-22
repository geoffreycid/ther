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

    # Summaries (add run{i} for each run)
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
    (h, w, c) = observation["image"].shape
    frames = dict_agent["frames"]
    # The network that predicts the mission only use the last frame
    keep_frames = frames * (c-1)
    # Number and name of actions
    n_actions = env.action_space.n
    action_names = [a.name for a in env.actions]

    # Number of types + colors + shades + sizes
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" and "SHADE_TO_IDX" and "SIZE_TO_IDX" in dict_env.keys():
        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())
        num_shades = len(dict_env["SHADE_TO_IDX"].keys())
        num_sizes = len(dict_env["SIZE_TO_IDX"].keys())
        num_token = num_colors + num_types + num_shades + num_sizes
    else:
        # The mission is not used
        num_token = 1

    # Load the model for the learned expert
    if use_learned_expert or use_expert_to_learn:
        if dict_expert["expert_type"] == "onehot":
            net_expert = models.PredMissionOneHot(c=c, frames=1, n_type=num_types, n_color=num_colors,
                                                  n_shade=num_shades, n_size=num_sizes, lr=dict_agent["lr"])

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

    if use_learned_expert:
        net_expert.load_state_dict(torch.load(dict_expert["expert_weights_path"]))
        net_expert.eval()

    # Define the agent
    if "dueling" in dict_agent["name"]:
        # Prioritized Experience Replay
        if "per" in dict_agent["name"]:
            agent = models.DuelingDoubleDQNPER
        else:
            agent = models.DuelingDoubleDQN
    else:
        # Prioritized Experience Replay
        if "per" in dict_agent["name"]:
            agent = models.DoubleDQNPER
        else:
            agent = models.DoubleDQN

    params = (h, w, c, n_actions, frames,
              dict_agent["lr"], num_token, device, dict_agent["use_memory"], dict_agent["use_text"])

    policy_net = agent(*params).to(device)
    target_net = agent(*params).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Replay memory
    if "per" in dict_agent["name"]:
        memory = replay_memory.PrioritizedReplayMemory(size=dict_agent["memory_size"],
                                                       seed=seed, alpha=dict_agent["alpha"], beta=dict_agent["beta"],
                                                       annealing_rate=dict_agent["annealing_rate"])
    else:
        memory = replay_memory.ReplayMemory(size=dict_agent["memory_size"], seed=seed)

    if use_expert_to_learn:
        memory_expert = replay_memory.ReplayMemoryExpert(size=dict_agent["memory_size"], seed=seed)

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
    correct_attributes_smoothing = [0] * (dict_env["n_attrib"] + 1)
    episode_done_carrying = 0

    # Logs sentences predicted by the expert
    with open(dict_agent["agent_dir"] + "/sentences.csv", "a") as log:
        log.write("{},{},{}\n".format("index", "prediction", "truth"))

    # Success rate useful for negative reward
    success_rate_smoothing = 0

    if not dict_env["wrong_object_terminal"]:
        wrong_object_picked = 0

    # Starting of the training procedure
    max_steps_reached = 0

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
        observation["image"] = observation["image"][:, :, :c]
        state_frames = [observation["image"]] * frames
        state_frames = np.concatenate(state_frames, axis=2).transpose((2, 0, 1))
        state_frames = torch.as_tensor(state_frames, dtype=torch.float32).unsqueeze(0)
        state["image"] = state_frames.to(device)

        for t in range(T_MAX):
            # Update the current state
            curr_state = state.copy()

            # Update epsilon
            epsilon = max(dict_agent["eps_init"] - steps_done * (dict_agent["eps_init"] - dict_agent["eps_final"])
                          / dict_expert["T_exploration"], dict_agent["eps_final"])

            # Select an action
            action = policy_net.select_action(curr_state, epsilon)

            # Interaction with the environment
            out_step = env.step(action)
            observation, reward, terminal, is_carrying = out_step[0], out_step[1], out_step[2], out_step[3]

            # Timeout
            if t == T_MAX - 1:
                terminal = True

            # Update of the next state
            observation_transposed \
                = torch.tensor(observation["image"].transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)

            state_image = torch.cat((curr_state["image"][:, c:], observation_transposed), dim=1)
            state["image"] = state_image

            # Update the number of steps
            steps_done += 1

            # Add a transition
            memory.add_transition(curr_state["image"], action, reward, state["image"], terminal, curr_state["mission"])
            # Useful for her, and learned expert
            if use_her or use_expert_to_learn:
                memory.store_transition(curr_state["image"], action, reward,
                                    state["image"], terminal, curr_state["mission"])

            if use_expert_to_learn and use_dense and action == 3 and not is_carrying\
                    and torch.equal(curr_state["image"][:, keep_frames:], state["image"][:, keep_frames:]):
                memory_expert.add_data_dense(curr_state=curr_state["image"][:, keep_frames:],
                                                    target=torch.tensor([0], dtype=torch.long).to(device))

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

            if use_expert_to_learn and memory_expert.len in dict_expert["update_network"]:
                start_use_expert, acc = net_expert.optimize_model(memory_expert, dict_expert["config_optimize_net"])
                # Not re-do the optimization
                dict_expert["update_network"] = dict_expert["update_network"][1:]
                # Decaying the number of iterations minimal before early stopping
                dict_expert["config_optimize_net"]["iterations_before_earlystopping"] \
                    = int(dict_expert["config_optimize_net"]["iterations_before_earlystopping"] / 1.1)
                # Accuracy
                writer.add_scalar("Accuracy", acc, global_step=memory_expert.len)

            if use_dense and use_expert_to_learn:
                if memory_expert.len_dense in dict_expert["update_network_dense"]:
                    if f1 < 0.99:
                        start_use_dense_expert, f1 \
                            = net_expert.optimize_model_dense(memory_expert, dict_expert["config_optimize_net_dense"])
                    # Not re-do the optimization
                    dict_expert["update_network_dense"] = dict_expert["update_network_dense"][1:]
                    # Decaying the number of iterations minimal before early stopping
                    dict_expert["config_optimize_net_dense"]["iterations_before_earlystopping"] \
                        = int(dict_expert["config_optimize_net_dense"]["iterations_before_earlystopping"] / 1.2)
                    # F1 score
                    writer.add_scalar("F1 score", f1, global_step=memory_expert.len_dense)

            # Update the target network
            if steps_done % dict_agent["update_target"] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Cumulative reward
            reward_smoothing += reward
            discounted_reward_smoothing += dict_agent["gamma"] ** t * reward

            # Record a trajectory
            if episode % dict_env["summary_trajectory"] == 0:
                utils.summary_trajectory(env, writer, n_actions, action_names,
                                         policy_net, state, observation, t, episode)
            # Summaries of Q values
            if "summary_mean_max_q" in dict_env.keys():
                if steps_done % dict_env["summary_mean_max_q"] == 0 and steps_done > dict_agent["batch_size"]:
                    utils.summary_mean_max_q(dict_agent, memory, policy_net, writer, steps_done)

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
                if is_carrying and reward > 0 and use_expert_to_learn:

                    expert_mission = net_expert.prediction_mission(curr_state["image"][:, keep_frames:]).to(device)

                    with open(dict_agent["agent_dir"] + "/sentences.csv", "a") as log:
                        log.write("{},{},{}\n".format(steps_done,
                                                      utils.sentences_from_indexes(expert_mission, idx2word),
                                                      utils.sentences_from_indexes(state["mission"], idx2word)))

                    correct_attributes = 0
                    for attribute in curr_state["mission"][-4:]:
                        if attribute in expert_mission:
                            correct_attributes += 1
                    correct_attributes_smoothing[correct_attributes] += 1

                    episode_done_carrying += 1

            if steps_done % dict_env["smoothing"] == 0:

                writer.add_scalar("Mean reward in {} steps".format(dict_env["smoothing"]),
                                  reward_smoothing / dict_env["smoothing"], global_step=steps_done)
                writer.add_scalar("Mean discounted reward in {} steps".format(dict_env["smoothing"]),
                                  discounted_reward_smoothing / dict_env["smoothing"], global_step=steps_done)
                writer.add_scalar("Mean length episode during {} steps".format(dict_env["smoothing"]),
                                  length_episode_done_smoothing / episode_done_smoothing, global_step=steps_done)
                writer.add_scalar("Mean reward per episode during {} steps".format(dict_env["smoothing"]),
                                  reward_smoothing / episode_done_smoothing, global_step=steps_done)
                writer.add_scalar("Timeout per episode during {} steps".format(dict_env["smoothing"]),
                                  timeout_smoothing / episode_done_smoothing, global_step=steps_done)
                writer.add_scalar("Object picked per episode during {} steps".format(dict_env["smoothing"]),
                                  object_picked_smoothing / episode_done_smoothing, global_step=steps_done)
                writer.add_scalar("Success rate per episode during {} steps".format(dict_env["smoothing"]),
                                  success_rate_smoothing / episode_done_smoothing, global_step=steps_done)

                if not dict_env["wrong_object_terminal"]:
                    writer.add_scalar("Wrong object picked rate per episode during {} steps".format(dict_env["smoothing"]),
                                      wrong_object_picked / episode_done_smoothing, global_step=steps_done)

                writer.add_scalar("length memory", len(memory), global_step=steps_done)
                writer.add_scalar("epsilon", epsilon, global_step=steps_done)

                episode_done_carrying = max(episode_done_carrying, 1)

                if use_expert_to_learn:
                    writer.add_scalar("length memory expert", memory_expert.len, global_step=steps_done)
                    writer.add_scalar("start use expert", start_use_expert, global_step=steps_done)

                    # Attributes predictions
                    writer.add_scalar("at least one attrib well predicted",
                                      sum(correct_attributes_smoothing[1:]) / episode_done_carrying,
                                      global_step=steps_done)
                    writer.add_scalar("at least two attrib well predicted",
                                      sum(correct_attributes_smoothing[2:]) / episode_done_carrying,
                                      global_step=steps_done)
                    writer.add_scalar("at least three attrib well predicted",
                                      sum(correct_attributes_smoothing[3:]) / episode_done_carrying,
                                      global_step=steps_done)
                    writer.add_scalar("at least 4 attrib well predicted",
                                      correct_attributes_smoothing[4] / episode_done_carrying,
                                      global_step=steps_done)

                if use_expert_to_learn and use_dense:
                    writer.add_scalar("length memory dense expert", memory_expert.len_dense, global_step=steps_done)
                    writer.add_scalar("start use dense expert", start_use_dense_expert, global_step=steps_done)

                # Re-init to 0
                reward_smoothing = 0
                discounted_reward_smoothing = 0
                episode_done_smoothing = 0
                length_episode_done_smoothing = 0
                timeout_smoothing = 0
                object_picked_smoothing = 0
                episode_done_carrying = 0
                success_rate_smoothing = 0
                correct_attributes_smoothing = [0] * (dict_env["n_attrib"] + 1)
                if not dict_env["wrong_object_terminal"]:
                    wrong_object_picked = 0

            # Save policy_net's parameters
            if steps_done % dict_env["save_model"] == 0:
                curr_path_to_save = os.path.join(path_save_model, "agent_model_steps_{}.pt".format(steps_done))
                torch.save(policy_net.state_dict(), curr_path_to_save)
                if use_expert_to_learn:
                    curr_path_to_save = os.path.join(path_save_model, "expert_model_steps_{}.pt".format(steps_done))
                    torch.save(net_expert.state_dict(), curr_path_to_save)

            # Save the replay buffer
            if steps_done % dict_env["save_replay_buffer"] == 0:
                path_save_replaybuffer = path_save_model + "/replaybuffer/"
                if not os.path.exists(path_save_replaybuffer):
                    os.makedirs(path_save_replaybuffer)
                with open(path_save_replaybuffer + "replaybuffer_steps_{}.pkl".format(steps_done), 'wb') as output:
                    dill.dump(memory, output, dill.HIGHEST_PROTOCOL)

            # Max steps reached
            if steps_done > dict_env["max_steps"] - 1:
                max_steps_reached = 1
                break

            # Episodes that end before timeout
            if terminal and t < T_MAX:

                # Fill the dataset to train the expert
                if use_expert_to_learn:

                    if use_dense:
                        memory_expert.add_data_dense(curr_state=curr_state["image"][:, keep_frames:],
                                                     target=torch.tensor([1], dtype=torch.long).to(device))

                    if dict_expert["expert_type"] in "onehot" + "dense" and reward > 0:
                        memory_expert.add_data(curr_state=curr_state["image"][:, keep_frames:],
                                               target=curr_state["mission"])

                    if dict_expert["expert_type"] in "rnn" and reward > 0:
                        miss = torch.cat((curr_state["mission"],
                                          dict_env["word2idx"]["END"]*torch.ones(1, device=device, dtype=torch.long)))
                        memory_expert.add_data(curr_state=curr_state["image"][:, keep_frames:],
                                               target=miss)

                # Re-labelling trajectories with the learned expert
                if start_use_expert and reward <= 0:
                    expert_reward = 1
                    with torch.no_grad():
                        expert_mission = net_expert.prediction_mission(curr_state["image"][:, keep_frames:]).to(
                            device)
                    memory.add_hindsight_transitions(reward=expert_reward, mission=expert_mission,
                                                     keep_last_transitions=dict_expert["keep_last_transitions"])
                    memory.erase_stored_transitions()

                # Re-labelling trajectories with HER
                if use_her:
                    hindsight_reward = out_step[4]
                    hindsight_target = out_step[5]
                    if use_noisy_her:
                        if dict_agent["use_text"]:
                            mission = utils.noisy_mission_one_threshold_text(hindsight_target,
                                                                             dict_env,
                                                                             dict_expert["parameters-noisy-her"])
                            mission = utils.indexes_from_sentences(utils.rnn_mission(mission, dict_env),
                                                                   dict_env["word2idx"]).to(device)
                        else:
                            mission = utils.noisy_mission_one_threshold(hindsight_target,
                                                      dict_env, dict_expert["parameters-noisy-her"]).to(device)
                    else:
                        if dict_agent["use_text"]:
                            mission = utils.indexes_from_sentences(utils.rnn_mission(hindsight_target, dict_env),
                                                                   dict_env["word2idx"]).to(device)
                        else:
                            mission = utils.mission_tokenizer(dict_env, target).to(device)

                    memory.add_hindsight_transitions(reward=hindsight_reward, mission=mission,
                                                     keep_last_transitions=dict_expert["keep_last_transitions"])
                    memory.erase_stored_transitions()

            if terminal:
                break

        if max_steps_reached:
            break

    env.close()
    writer.close()
