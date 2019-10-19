import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import numpy as np


def summary_trajectory(env, writer, n_actions, action_names, policy_net, state, observation, t, episode):
    image = env.render("rgb_array")
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("Actions")
    if policy_net.use_text:
        copy_state = state.copy()
        copy_state["text_length"] = [state["mission"].shape[0]]
        copy_state["mission"] = state["mission"].unsqueeze(0)
        ax1.bar(range(n_actions), policy_net(copy_state).cpu().detach().numpy().reshape(-1))
    else:
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
    copy_state["text_length"] = copy_state["mission"].shape[0]
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
    if "per" in dict_agent["agent"]:
        transitions, is_weights, transition_idxs = memory.sample(dict_agent["batch_size"])
    else:
        transitions = memory.sample(dict_agent["batch_size"])
    batch_transitions = memory.transition(*zip(*transitions))
    batch_curr_state = torch.cat(batch_transitions.curr_state)
    if policy_net.use_text:
        text_length = [None] * dict_agent["batch_size"]
        for ind, mission in enumerate(batch_transitions.mission):
            text_length[ind] = mission.shape[0]
        batch_text_length = torch.tensor(text_length, dtype=torch.long).to(policy_net.device)
        batch_mission = nn.utils.rnn.pad_sequence(batch_transitions.mission, batch_first=True).to(policy_net.device)
    else:
        batch_mission = torch.cat(batch_transitions.mission)

    # Compute targets according to the Bellman eq
    if policy_net.use_text:
        batch_curr_state_dict = {
            "image": batch_curr_state,
            "mission": batch_mission,
            "text_length": batch_text_length
        }
    else:
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
    num_shade = len(dict_env["SHADE_TO_IDX"].keys())
    num_size = len(dict_env["SIZE_TO_IDX"].keys())

    mission_color_onehot = torch.zeros(num_colors)
    if num_colors > 1:
        mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1

    mission_type_onehot = torch.zeros(num_types)
    mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

    mission_shade_onehot = torch.zeros(num_shade)
    mission_shade_onehot[dict_env["SHADE_TO_IDX"][target["shade"]]] = 1

    mission_size_onehot = torch.zeros(num_size)
    mission_size_onehot[dict_env["SIZE_TO_IDX"][target["size"]]] = 1

    return torch.cat((mission_type_onehot, mission_color_onehot, mission_shade_onehot, mission_size_onehot)).unsqueeze(0)


def indexes_from_sentences(mission, word2idx):
    words = mission.split()
    indexes = []
    for word in words:
        indexes.append(word2idx[word])
    return torch.LongTensor(indexes)


def sentences_from_indexes(mission, idx2word):
    sentence = ''
    for idx in mission:
        sentence += idx2word[idx.item()] + " "
    return sentence[:-1]


def rnn_mission(target, dict_env):

    if dict_env["shuffle_attrib"]:
        attrib = [target["size"], target["shade"], target["color"]]
        random.shuffle(attrib)
        miss = tuple(attrib) + (target["type"],)
        descStr = '%s %s %s %s' % miss
    else:
        descStr = '%s %s %s %s' % (target["size"], target["shade"], target["color"], target["type"])

    # Generate the mission string
    idx = random.randint(0, 4)
    if idx == 0:
        mission = 'get a %s' % descStr
    elif idx == 1:
        mission = 'go get a %s' % descStr
    elif idx == 2:
        mission = 'fetch a %s' % descStr
    elif idx == 3:
        mission = 'go fetch a %s' % descStr
    elif idx == 4:
        mission = 'you must fetch a %s' % descStr

    return mission


def noisy_mission(target, dict_env, config):

    num_colors = len(dict_env["COLOR_TO_IDX"].keys())
    num_types = len(dict_env["TYPE_TO_IDX"].keys())
    num_shade = len(dict_env["SENIORITY_TO_IDX"].keys())
    num_size = len(dict_env["SIZE_TO_IDX"].keys())

    mission_color_onehot = torch.zeros(num_colors)
    if num_colors > 1:
        mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1

    mission_type_onehot = torch.zeros(num_types)
    mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

    mission_shade_onehot = torch.zeros(num_shade)
    mission_shade_onehot[dict_env["SHADE_TO_IDX"][target["shade"]]] = 1

    mission_size_onehot = torch.zeros(num_size)
    mission_size_onehot[dict_env["SIZE_TO_IDX"][target["size"]]] = 1

    proba = random.random()
    if proba < config["proba-noisy"]:

        if "mask" in config["noisy-type"]:
            if "color" in config["attrib"]:
                mission_color_onehot = torch.zeros(num_colors)
            if "type" in config["attrib"]:
                mission_type_onehot = torch.zeros(num_types)
            if "shade" in config["attrib"]:
                mission_shade_onehot = torch.zeros(num_shade)
            if "size" in config["attrib"]:
                mission_size_onehot = torch.zeros(num_size)
            if "random" in config["attrib"]:
                attrib = random.randint(0, 3)
                if attrib == 0:
                    mission_color_onehot = torch.zeros(num_colors)
                elif attrib == 1:
                    mission_type_onehot = torch.zeros(num_types)
                elif attrib == 2:
                    mission_shade_onehot = torch.zeros(num_shade)
                elif attrib == 3:
                    mission_size_onehot = torch.zeros(num_size)

        if "noise" in config["noisy-type"]:
            if "color" in config["attrib"]:
                idx = random.randint(0, num_colors - 1)
                mission_color_onehot[idx] = 1
            if "type" in config["attrib"]:
                idx = random.randint(0, num_types - 1)
                mission_type_onehot[idx] = 1
            if "shade" in config["attrib"]:
                idx = random.randint(0, num_shade - 1)
                mission_shade_onehot[idx] = 1
            if "size" in config["attrib"]:
                idx = random.randint(0, num_size - 1)
                mission_size_onehot[idx] = 1
            if "random" in config["attrib"]:
                attrib = random.randint(0, 3)
                if attrib == 0:
                    idx = random.randint(0, num_colors - 1)
                    mission_color_onehot[idx] = 1
                elif attrib == 1:
                    idx = random.randint(0, num_types - 1)
                    mission_type_onehot[idx] = 1
                elif attrib == 2:
                    idx = random.randint(0, num_shade - 1)
                    mission_shade_onehot[idx] = 1
                elif attrib == 3:
                    idx = random.randint(0, num_size - 1)
                    mission_size_onehot[idx] = 1

    return torch.cat((mission_type_onehot, mission_color_onehot, mission_shade_onehot, mission_size_onehot)).unsqueeze(0)


def noisy_mission_one_threshold(target, dict_env, config):

    num_colors = len(dict_env["COLOR_TO_IDX"].keys())
    num_types = len(dict_env["TYPE_TO_IDX"].keys())
    num_shade = len(dict_env["SHADE_TO_IDX"].keys())
    num_size = len(dict_env["SIZE_TO_IDX"].keys())

    mission_color_onehot = torch.zeros(num_colors)
    if num_colors > 1:
        mission_color_onehot[dict_env["COLOR_TO_IDX"][target["color"]]] = 1

    mission_type_onehot = torch.zeros(num_types)
    mission_type_onehot[dict_env["TYPE_TO_IDX"][target["type"]]] = 1

    mission_shade_onehot = torch.zeros(num_shade)
    mission_shade_onehot[dict_env["SHADE_TO_IDX"][target["shade"]]] = 1

    mission_size_onehot = torch.zeros(num_size)
    mission_size_onehot[dict_env["SIZE_TO_IDX"][target["size"]]] = 1

    proba_type = random.random()
    if proba_type < config["proba-noisy"]:
        idx = random.randint(0, num_types - 1)
        mission_type_onehot[idx] = 1

    proba_color = random.random()
    if proba_color < config["proba-noisy"]:
        idx = random.randint(0, num_colors - 1)
        mission_color_onehot[idx] = 1

    proba_shade = random.random()
    if proba_shade < config["proba-noisy"]:
        idx = random.randint(0, num_shade - 1)
        mission_shade_onehot[idx] = 1

    proba_size = random.random()
    if proba_size < config["proba-noisy"]:
        idx = random.randint(0, num_size - 1)
        mission_size_onehot[idx] = 1

    return torch.cat((mission_type_onehot, mission_color_onehot, mission_shade_onehot, mission_size_onehot)).unsqueeze(0)


def noisy_mission_one_threshold_text(target, dict_env, config):

    colors = list(dict_env["COLOR_TO_IDX"].keys())
    types = list(dict_env["TYPE_TO_IDX"].keys())
    shades = list(dict_env["SHADE_TO_IDX"].keys())
    sizes = list(dict_env["SIZE_TO_IDX"].keys())

    proba_type = random.random()
    if proba_type < config["proba-noisy"]:
        target["type"] = random.choice(types)

    proba_color = random.random()
    if proba_color < config["proba-noisy"]:
        target["color"] = random.choice(colors)

    proba_shade = random.random()
    if proba_shade < config["proba-noisy"]:
        target["shade"] = random.choice(shades)

    proba_size = random.random()
    if proba_size < config["proba-noisy"]:
        target["size"] = random.choice(sizes)

    return target


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
    mission_seniority_onehot[dict_env["SENIORITY_TO_IDX"][target["shade"]]] = 1

    mission_size_onehot = np.zeros(num_size)
    mission_size_onehot[dict_env["SIZE_TO_IDX"][target["size"]]] = 1
    mission = np.concatenate((mission_type_onehot, mission_color_onehot, mission_seniority_onehot, mission_size_onehot))
    return np.expand_dims(mission, axis=0)


def mission_tokenizer_onehot(dict_env, target, num_token):
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():

        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())

        mission_onehot = torch.zeros(num_token)
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


