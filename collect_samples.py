import os

import numpy as np
import gym_minigrid.envs.game as game
import torch

import summaryutils as utils
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
        attrib = [target["size"], target["seniority"], target["color"]]
        random.shuffle(attrib)
        miss = tuple(attrib) + (target["type"],)
        descStr = '%s %s %s %s' % miss
    else:
        descStr = '%s %s %s %s' % (target["size"], target["seniority"], target["color"], target["type"])

    # Generate the mission string
    idx = random.randint(0, 4)
    if idx == 0:
        mission = 'get a %s .' % descStr
    elif idx == 1:
        mission = 'go get a %s .' % descStr
    elif idx == 2:
        mission = 'fetch a %s .' % descStr
    elif idx == 3:
        mission = 'go fetch a %s .' % descStr
    elif idx == 4:
        mission = 'you must fetch a %s .' % descStr

    return mission


def collect_samples(dict_env, dict_agent, use_her, use_imc, use_dense=0, use_rnn=0):

    assert use_her != use_imc, "Can't use both use_her and use_imc!"

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

    # Number of types + colors
    if "COLOR_TO_IDX" and "TYPE_TO_IDX" in dict_env.keys():
        num_colors = len(dict_env["COLOR_TO_IDX"].keys())
        num_types = len(dict_env["TYPE_TO_IDX"].keys())
        dim_tokenizer = num_colors * num_types
    else:
        # The mission is not used
        dim_tokenizer = 1

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
            "seniority": env.targetSeniority,
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
    return memory_collectsample


if __name__ == "__main__":

    dict_env = {
        "name": "10x10-C4-N2-O8",
        "device": "cpu",
        "game_type": "fetch",
        "wrong_object_terminal": 1,
        "reward_if_wrong_object": 0,
        "use_held_out_mission": 0,
        "shuffle_attrib": 1,
        "size": 10,
        "numObjs": 8,
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
        "SENIORITY_TO_IDX": {
            "veryyoung": 0,
            "young": 1,
            "middle": 2,
            "old": 3,
            "veryold": 4
        },
        "SIZE_TO_IDX": {
            "verysmall": 0,
            "small": 1,
            "average": 2,
            "big": 3,
            "verybig": 4
        },
        "T_max": 50,

        "word2idx": {
            "pad": 0,
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
            "veryyoung": 15,
            "young": 16,
            "middle": 17,
            "old": 18,
            "veryold": 19,
            "verysmall": 20,
            "small": 21,
            "average": 22,
            "big": 23,
            "verybig": 24,
            "start": 25,
            ".": 26
        }
    }

    dict_agent = {
        "name": "double-dqn",
        "agent": "double-dqn",
        "seed": 29,
        "frames": 4,
        "n_keep_correspondence": 1,
        "skew_ratio": 0.5,
        "memory_size": 110000,
        "use_her": 1,
    }

    if dict_agent["use_her"]:
        use_her = 1
        use_imc = 0
    else:
        use_her = 0
        use_imc = 1
    mem = collect_samples(dict_env, dict_agent, use_her=use_her, use_imc=use_imc, use_dense=0, use_rnn=1)

    with open("/home/gcideron/datasets/collect_samples_{}_memory_size_{}_frames_{}_missions_her_cpu_rnn_shuffle_attrib.pkl"
                      .format(int(dict_agent["memory_size"]), dict_agent["frames"], 300), 'wb') as output:
        dill.dump(mem, output, dill.HIGHEST_PROTOCOL)