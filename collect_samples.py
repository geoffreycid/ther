import os

import numpy as np
import gym_minigrid.envs.game as game
import torch

import utils as utils
import collections
import random
import operator
import dill
import time
"""training procedure"""


class CollectSampleMemory(object):
    def __init__(self, size, n_keep_correspondence, skew_ratio, seed):
        self.imc = collections.namedtuple("ImMis",
                                                 ["state", "mission", "target"])
        self.memory_size = int(size * n_keep_correspondence)
        self.position = 0
        self.len = 0
        self.memory = [None for _ in range(self.memory_size)]
        self.stored_data = []
        self.list_of_targets = []
        self.skew_ratio = skew_ratio
        self.episodes_done = 0
        self.memory_dense = [None for _ in range(self.memory_size * 5)]
        self.position_dense = 0
        self.imc_dense = collections.namedtuple("dense", ["state", "target"])
        self.stored_dense_data = []
        random.seed(seed)

    def add_data(self, curr_state, mission, target):
        self.memory[self.position] = self.imc(curr_state, mission, target)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0

    def add_data_dense(self, curr_state, target):
        if self.position_dense > self.memory_size * 5 - 1:
            return
        self.memory_dense[self.position_dense] = self.imc_dense(curr_state, target)
        self.position_dense += 1

    def store_dense_data(self, curr_state, target):
        self.stored_dense_data.append(self.imc_dense(curr_state, target))

    def add_stored_dense_data(self, target):
        if self.position_dense > self.memory_size * 3 - 1:
            return
        if len(self.stored_dense_data) > 2:
            self.memory_dense[self.position_dense] = self.stored_dense_data[-3]
            self.position_dense = min(self.memory_size * 3 - 1, self.position_dense + 1)
        self.memory_dense[self.position_dense] = self.stored_dense_data[-1]._replace(target=target)
        self.position_dense += 1
        self.stored_dense_data = []

    def sample(self, batch_size):
        #if len(self.memory) < batch_size:
        #    return random.sample(self.memory, batch_size)
        #else:
        #    np_targets = np.array(self.list_of_targets, dtype=np.float)
        #    np_targets[np_targets == 1] = self.skew_ratio / sum(np_targets == 1)
        #    np_targets[np_targets == 0] = (1-self.skew_ratio) / sum(np_targets == 0)
        #    sampled_idxs = np.random.choice(np.arange(len(np_targets)),
        #                                       size=batch_size, replace=False, p=np_targets)
        #    op = operator.itemgetter(*sampled_idxs)
        #    return op(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return self.len

    def store_data(self, curr_state, mission, target):
        self.stored_data.append(self.imc(curr_state, mission, target))

    def add_stored_data(self, target, n_keep_correspondence):
        length = len(self.stored_data)
        n = min(length, n_keep_correspondence)
        for i in range(length-1, length-1-n, -1):
            self.memory[self.position] = self.stored_data[i]._replace(target=target)
            # Update the position and the len of the memory size
            self.position += 1
            self.len = min(self.memory_size, self.len + 1)
            if self.position > self.memory_size - 1:
                self.position = 0
        self.erase_stored_data()

    def erase_stored_data(self):
        self.stored_data = []


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


def collect_samples(dict_env, dict_agent, use_her, use_imc, use_dense=0, use_rnn=0):

    assert use_her != use_imc, "Can't use both use_her and use_imc!"

    cum_reward = 0
    rewards_at_steps = []

    # Device to use
    if "device" in dict_env:
        device = dict_env["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment
    env = game.game(dict_env)

    # Fix all seeds
    seed = dict_agent["seed"]
    env.seed(seed)
    np.random.seed(seed)

    # Number of types + colors
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():
        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())
        num_token = num_colors * num_types
    else:
        # The mission is not used
        num_token = 1

    # Memory that collect all the samples
    memory_collectsample = CollectSampleMemory(size=dict_agent["memory_size"],
                                               n_keep_correspondence=dict_agent["n_keep_correspondence"],
                                               skew_ratio=dict_agent["skew_ratio"], seed=seed)
    if not use_dense:
        memory_collectsample.memory_dense = []

    # Max steps per episode
    T_MAX = min(dict_env["T_max"], env.max_steps)

    # Number of times the agent interacted with the environment
    steps_done = 0
    episode = 0

    # Starting of the training procedure
    max_steps_reached = 0

    init_time = time.time()
    while True:
        # New episode
        episode += 1
        state = {}
        observation = env.reset()
        # Erase stored transitions
        memory_collectsample.erase_stored_data()

        # One hot encoding of the type and the color of the target object
        target = {
            "color": env.targetColor,
            "type": env.targetType,
            "shade": env.targetShade,
            "size": env.targetSize
        }
        state["mission"] = utils.mission_tokenizer(dict_env, target).to(device)

        # Stacking frames to make a state
        # Only keep the first 4 channels, the fifth one is a flag for open/closed door
        observation["image"] = observation["image"][:, :, :4]
        state_frames = [observation["image"]] * dict_agent["frames"]
        state_frames = np.concatenate(state_frames, axis=2).transpose((2, 0, 1))
        state_frames = torch.as_tensor(state_frames, dtype=torch.float32).unsqueeze(0)
        state["image"] = state_frames.to(device)

        for t in range(T_MAX):
            if steps_done % 50000 == 0:
                print("steps: {}, time: {}, len memory: {}, len memory dense {}"
                      .format(steps_done, round(time.time()-init_time, 2), len(memory_collectsample),
                              memory_collectsample.position_dense))
            # Update the current state
            curr_state = state.copy()

            # Select an action
            action = env.action_space.sample()

            # Interaction with the environment
            out_step = env.step(action)
            observation, reward, terminal, return_her, is_carrying \
                = out_step[0], out_step[1], out_step[2], out_step[3], out_step[4]

            observation["image"] = observation["image"][:, :, :4]
            observation_prep \
                = torch.as_tensor(observation['image'].transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)

            state_frames = torch.cat((curr_state["image"][:, 4:], observation_prep), dim=1)
            state["image"] = state_frames

            if reward > 0:
                cum_reward += 1
            if cum_reward % 1000 == 0 and cum_reward > 0:
                print("cum rewards {} in {} steps".format(cum_reward, steps_done))

            # If pickup action does not change the state => no object in front of the agent
            if use_dense and action == 3 and torch.equal(curr_state["image"][:, 12:], state["image"][:, 12:]):
                memory_collectsample.add_data_dense(curr_state=curr_state["image"],
                                                    target=torch.tensor([0], dtype=torch.long).to(device))

            # Update the number of steps
            steps_done += 1

            # Add transition
            memory_collectsample.store_data(curr_state["image"], curr_state["mission"], 0)
            #memory_collectsample.store_dense_data(curr_state["image"], torch.tensor([0], dtype=torch.long).to(device))

            if len(memory_collectsample) == memory_collectsample.memory_size:
                max_steps_reached = 1
                break
            # Terminate the episode if terminal state
            if terminal:
                if is_carrying:
                    if use_dense:
                        memory_collectsample.add_data_dense(curr_state=curr_state["image"],
                                                            target=torch.tensor([1], dtype=torch.long).to(device))
                    if use_imc:
                        if reward == 0:
                            target_imc = torch.tensor([0], dtype=torch.long).to(device)
                        else:
                            target_imc = torch.tensor([1], dtype=torch.long).to(device)

                        memory_collectsample.add_stored_data(target_imc, dict_agent["n_keep_correspondence"])

                    elif use_her:
                        if reward == 0:
                            if return_her:
                                hindsight_reward = out_step[5]
                                hindsight_target = out_step[6]
                                if use_rnn:
                                    target_mission = utils.indexes_from_sentences(rnn_mission(hindsight_target, dict_env),
                                                                                  dict_env["word2idx"])
                                else:
                                    target_mission = utils.mission_tokenizer(dict_env, hindsight_target).to(device)

                                memory_collectsample.add_stored_data(target_mission,
                                                                     dict_agent["n_keep_correspondence"])
                        else:
                            if use_rnn:
                                target_mission = utils.indexes_from_sentences(rnn_mission(target, dict_env),
                                                                              dict_env["word2idx"])
                                memory_collectsample.add_stored_data(target_mission,
                                                                     dict_agent["n_keep_correspondence"])

                            else:
                                memory_collectsample.add_stored_data(state["mission"],
                                                                     dict_agent["n_keep_correspondence"])
                break

        if max_steps_reached:
            memory_collectsample.episodes_done = episode
            break
        env.close()
    print("num steps", steps_done)
    print("num ep", episode)
    print("rewards at steps", rewards_at_steps[:10])
    return memory_collectsample


if __name__ == "__main__":

    dict_env = {
        "name": "12x12-C4-N2-O10",
        "device": "cpu",
        "game_type": "fetch",
        "wrong_object_terminal": 1,
        "reward_if_wrong_object": 0,
        "use_defined_missions": 1,
        "shuffle_attrib": 1,
        "size": 12,
        "numObjs": 10,
        "manual": 0,
        "random_target": 1,
        "oneobject": 0,
        "COLOR_TO_IDX": {
            "red": 0,
            "green": 1,
            "blue": 2,
            "purple": 3,
            "yellow": 4,
            "grey": 5
        },
        "TYPE_TO_IDX": {
            "key": 0,
            "ball": 1
        },
        "SHADE_TO_IDX": {
            "very_light": 0,
            "light": 1,
            "neutral": 2,
            "dark": 3,
            "very_dark": 4
        },
        "SIZE_TO_IDX": {
            "tiny": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
            "giant": 4
        },
        "T_max": 50,

        "word2idx": {
            "PAD": 0,
            "get": 1,
            "a": 2,
            "go": 3,
            "fetch": 4,
            "you": 5,
            "must": 6,
            "red": 7,
            "green": 8,
            "blue": 9,
            "purple": 10,
            "yellow": 11,
            "grey": 12,
            "key": 13,
            "ball": 14,
            "very_light": 15,
            "light": 16,
            "neutral": 17,
            "dark": 18,
            "very_dark": 19,
            "tiny": 20,
            "small": 21,
            "medium": 22,
            "large": 23,
            "giant": 24,
            "BEG": 25,
            "END": 26
        },

        "missions": [["purple", "ball", "giant", "light"],
                     ["purple", "key", "medium", "light"],
                     ["red", "ball", "small", "very_dark"],
                     ["yellow", "ball", "medium", "light"],
                     ["red", "ball", "giant", "very_light"],
                     ["blue", "key", "medium", "neutral"],
                     ["yellow", "key", "small", "neutral"],
                     ["purple", "key", "small", "light"],
                     ["purple", "ball", "giant", "very_dark"],
                     ["yellow", "ball", "large", "dark"],
                     ["blue", "key", "large", "light"],
                     ["grey", "key", "small", "very_dark"],
                     ["green", "key", "giant", "very_light"],
                     ["yellow", "key", "small", "very_light"],
                     ["blue", "key", "large", "very_dark"],
                     ["red", "key", "medium", "dark"],
                     ["red", "ball", "large", "dark"],
                     ["yellow", "key", "giant", "dark"],
                     ["grey", "key", "tiny", "dark"],
                     ["red", "key", "giant", "light"],
                     ["yellow", "key", "large", "dark"],
                     ["blue", "ball", "small", "neutral"],
                     ["red", "key", "small", "very_dark"],
                     ["green", "ball", "small", "dark"],
                     ["yellow", "key", "giant", "neutral"],
                     ["blue", "key", "giant", "very_dark"],
                     ["purple", "ball", "small", "dark"],
                     ["red", "key", "medium", "very_dark"],
                     ["grey", "key", "giant", "very_dark"],
                     ["yellow", "key", "small", "dark"],
                     ["grey", "key", "large", "light"],
                     ["purple", "key", "large", "dark"],
                     ["purple", "ball", "medium", "light"],
                     ["green", "key", "giant", "very_dark"],
                     ["green", "ball", "large", "light"],
                     ["grey", "ball", "medium", "neutral"],
                     ["purple", "ball", "giant", "dark"],
                     ["purple", "ball", "tiny", "light"],
                     ["blue", "key", "medium", "dark"],
                     ["green", "key", "medium", "dark"],
                     ["purple", "ball", "medium", "neutral"],
                     ["yellow", "key", "small", "light"],
                     ["red", "key", "tiny", "neutral"],
                     ["grey", "ball", "tiny", "light"],
                     ["purple", "ball", "small", "very_light"],
                     ["purple", "key", "large", "neutral"],
                     ["blue", "ball", "tiny", "light"],
                     ["purple", "ball", "tiny", "very_light"],
                     ["yellow", "ball", "giant", "neutral"],
                     ["grey", "ball", "medium", "very_dark"],
                     ["green", "key", "large", "neutral"],
                     ["red", "ball", "medium", "dark"],
                     ["blue", "key", "medium", "very_light"],
                     ["red", "ball", "giant", "dark"],
                     ["yellow", "ball", "large", "very_dark"],
                     ["grey", "key", "small", "very_light"],
                     ["green", "ball", "large", "dark"],
                     ["purple", "key", "large", "very_dark"],
                     ["grey", "ball", "small", "very_dark"],
                     ["red", "ball", "giant", "neutral"]]
    }

    dict_agent = {
        "name": "double-dqn",
        "agent": "double-dqn",
        "seed": 1,
        "frames": 4,
        "n_keep_correspondence": 1,
        "skew_ratio": 0.5,
        "memory_size": 20000,
        "use_her": 1,
    }

    if dict_agent["use_her"]:
        use_her = 1
        use_imc = 0
    else:
        use_her = 0
        use_imc = 1

    mem = collect_samples(dict_env, dict_agent, use_her=use_her, use_imc=use_imc, use_dense=0, use_rnn=1)

    with open("/home/user/datasets/test_memory_size_{}_seed_{}_60_missions.pkl"
                      .format(int(dict_agent["memory_size"]), int(dict_agent["seed"])), 'wb') as output:
        dill.dump(mem, output, dill.HIGHEST_PROTOCOL)

    # /home/user/datasets/collect_samples_{}_memory_size_{}_frames_{}_missions_her_cpu_rnn_shuffle_attrib
    # .format(int(dict_agent["memory_size"]), dict_agent["frames"], 300), 'wb') as output: