import matplotlib.pyplot as plt
import torch
import numpy as np


def summary_trajectory(env, dict_env, writer, n_actions, action_names, policy_net, state, observation, t, episode, use_imc):
        # image = env.render("rgb_array")
        # fig = plt.figure(figsize=(10, 6))
        # ax1 = fig.add_subplot(2, 2, 2)
        # ax1.set_title("Actions")
        # ax1.bar(range(n_actions), policy_net(state).cpu().detach().numpy().reshape(-1))
        # ax1.set_xticks(range(n_actions))
        # ax1.set_xticklabels(action_names, fontdict=None, minor=False)
        # ax1.set_ylabel("Q values")
        #
        # ax2 = fig.add_subplot(2, 2, 1)
        # ax2.set_title("{}".format(observation["mission"]))
        # ax2.imshow(image)
        #
        # proba_type, proba_color = policy_net.pred_proba_mission(state)
        # proba_type, proba_color = proba_type.cpu().numpy().reshape(-1), proba_color.cpu().numpy().reshape(-1)
        # type_key_sorted = [kv[0] for kv in sorted(dict_env["TYPE_TO_IDX"].items(), key=lambda kv:(kv[1], kv[0]))]
        # color_key_sorted = [kv[0] for kv in sorted(dict_env["COLOR_TO_IDX"].items(), key=lambda kv: (kv[1], kv[0]))]
        #
        # ax3 = fig.add_subplot(2, 2, 3)
        # ax3.set_title("pred mission type")
        # ax3.bar(range(proba_type.shape[0]), proba_type)
        # ax3.set_xticks(range(proba_type.shape[0]))
        # ax3.set_xticklabels(type_key_sorted, fontdict=None, minor=False)
        # ax3.set_ylabel("Probabilities")
        #
        # ax4 = fig.add_subplot(2, 2, 4)
        # ax4.set_title("pred mission color")
        # ax4.bar(range(proba_color.shape[0]), proba_color)
        # ax4.set_xticks(range(proba_color.shape[0]))
        # ax4.set_xticklabels(color_key_sorted, fontdict=None, minor=False)
        # ax4.set_ylabel("Probabilities")
        #
        # writer.add_figure("Q values pred mission episode {}".format(episode), fig, global_step=t)
    image = env.render("rgb_array")
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Actions")
    ax1.bar(range(n_actions), policy_net(state).cpu().detach().numpy().reshape(-1))
    ax1.set_xticks(range(n_actions))
    ax1.set_xticklabels(action_names, fontdict=None, minor=False)
    ax1.set_ylabel("Q values")
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.set_title("{}".format(observation["mission"]))
    ax2.imshow(image)
    writer.add_figure("Q values episode {}".format(episode + 1), fig, global_step=t)


def summary_trajectory_numpy(env, writer, n_actions, action_names, policy_net, state, observation, t, episode):
    copy_state = state.copy()
    copy_state["image"] = torch.from_numpy(state["image"]).to(policy_net.device).float()
    copy_state["mission"] = torch.from_numpy(state["mission"]).to(policy_net.device).float()
    image = env.render("rgb_array")
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Actions")
    ax1.bar(range(n_actions), policy_net(state).cpu().detach().numpy().reshape(-1))
    ax1.set_xticks(range(n_actions))
    ax1.set_xticklabels(action_names, fontdict=None, minor=False)
    ax1.set_ylabel("Q values")
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.set_title("{}".format(observation["mission"]))
    ax2.imshow(image)
    writer.add_figure("Q values episode {}".format(episode + 1), fig, global_step=t)


def summary_mean_max_q(dict_agent, memory, policy_net, writer, steps_done):
    # Sample a batch of states and compute mean and max values of Q
    if dict_agent["agent"] == "dqn-per" or dict_agent["agent"] == "double-dqn-per":
        transitions, is_weights, transition_idxs = memory.sample(dict_agent["batch_size"])
    else:
        transitions = memory.sample(dict_agent["batch_size"])
    batch_transitions = memory.transition(*zip(*transitions))
    batch_curr_state = torch.cat(batch_transitions.curr_state)
    batch_mission = torch.cat(batch_transitions.mission)
    batch_curr_state_dict = {
        "image": batch_curr_state,
        "mission": batch_mission
    }
    q_values = policy_net(batch_curr_state_dict).detach()
    writer.add_scalar("mean Q", torch.mean(q_values).cpu().numpy().reshape(-1)
                      , global_step=steps_done)
    writer.add_scalar("max Q", torch.max(q_values).cpu().numpy().reshape(-1)
                      , global_step=steps_done)


def summary_mean_max_q_numpy(dict_agent, memory, policy_net, writer, steps_done):
    # Sample a batch of states and compute mean and max values of Q
    if dict_agent["agent"] == "dqn-per" or dict_agent["agent"] == "double-dqn-per":
        transitions, is_weights, transition_idxs = memory.sample(dict_agent["batch_size"])
    else:
        transitions = memory.sample(dict_agent["batch_size"])

    batch_transitions = memory.transition(*zip(*transitions))
    batch_curr_state = torch.from_numpy(np.concatenate(batch_transitions.curr_state)).to(policy_net.device).float()
    batch_mission = torch.from_numpy(np.concatenate(batch_transitions.mission)).to(policy_net.device).float()

    batch_curr_state_dict = {
        "image": batch_curr_state,
        "mission": batch_mission
    }
    q_values = policy_net(batch_curr_state_dict).detach()
    writer.add_scalar("mean Q", torch.mean(q_values).cpu().numpy().reshape(-1)
                      , global_step=steps_done)
    writer.add_scalar("max Q", torch.max(q_values).cpu().numpy().reshape(-1)
                      , global_step=steps_done)


def mission_tokenizer(dict_env, target):

    num_colors = len(dict_env["COLOR_TO_IDX"].keys())
    num_types = len(dict_env["TYPE_TO_IDX"].keys())
    num_seniority = len(dict_env["SENIORITY_TO_IDX"].keys())
    num_size = len(dict_env["SIZE_TO_IDX"].keys())

    mission_color_onehot = torch.zeros(num_colors)
    if num_colors > 1:
        mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1

    mission_type_onehot = torch.zeros(num_types)
    mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

    mission_seniority_onehot = torch.zeros(num_seniority)
    mission_seniority_onehot[dict_env["SENIORITY_TO_IDX"][target["seniority"]]] = 1

    mission_size_onehot = torch.zeros(num_size)
    mission_size_onehot[dict_env["SIZE_TO_IDX"][target["size"]]] = 1

    return torch.cat((mission_type_onehot, mission_color_onehot, mission_seniority_onehot, mission_size_onehot)).unsqueeze(0)


def noisy_misison(mission, dict_expert):
    parameters = dict_expert["noisy_her_parameters"]
    if parameters["mask"] > 0:

        pass
    pass


def mission_tokenizer_numpy(dict_env, target):

    num_colors = len(dict_env["COLOR_TO_IDX"].keys())
    num_types = len(dict_env["TYPE_TO_IDX"].keys())
    num_seniority = len(dict_env["SENIORITY_TO_IDX"].keys())
    num_size = len(dict_env["SIZE_TO_IDX"].keys())

    mission_color_onehot = np.zeros(num_colors)
    if num_colors > 1:
        mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1

    mission_type_onehot = np.zeros(num_types)
    mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

    mission_seniority_onehot = np.zeros(num_seniority)
    mission_seniority_onehot[dict_env["SENIORITY_TO_IDX"][target["seniority"]]] = 1

    mission_size_onehot = np.zeros(num_size)
    mission_size_onehot[dict_env["SIZE_TO_IDX"][target["size"]]] = 1
    mission = np.concatenate((mission_type_onehot, mission_color_onehot, mission_seniority_onehot, mission_size_onehot))
    return np.expand_dims(mission, axis=0)


def mission_tokenizer_onehot(dict_env, target, dim_tokenizer):
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():

        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())

        mission_onehot = torch.zeros(dim_tokenizer)
        indice = dict_env["TYPE_TO_IDX"][target["type"]] * num_colors + dict_env["COLOR_TO_IDX"][target["color"]]
        mission_onehot[indice] = 1

        return mission_onehot


def mission_tokenizer_separate(dict_env, target):

    num_colors = len(dict_env["COLOR_TO_IDX"].keys())
    num_types = len(dict_env["TYPE_TO_IDX"].keys())

    mission_color_onehot = torch.zeros(num_colors)
    if num_colors > 1:
        mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1
    mission_type_onehot = torch.zeros(num_types)
    mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

    return mission_type_onehot.unsqueeze(0), mission_color_onehot.unsqueeze(0)


