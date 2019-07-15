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
        random.seed(seed)

    def add_data(self, curr_state, mission, target):
        self.memory[self.position] = self.imc(curr_state, mission, target)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0

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


def collect_samples(dict_env, dict_agent, use_her, use_imc):

    assert use_her != use_imc, "Can't use both use_her and use_imc!"

    # Device to use
    if "device" in dict_env:
        device = dict_env["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment
    env = game.game(dict_env, dict_agent)

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
            "type": env.targetType
        }
        state["mission"] = utils.mission_tokenizer(dict_env, target).to(device)

        # Stacking frames to make a state
        # Only keep the first 2 channels, the third one is a flag for open/closed door
        observation["image"] = observation["image"][:, :, :2]
        state_frames = [observation["image"]] * dict_agent["frames"]
        state_frames = np.concatenate(state_frames, axis=2).transpose((2, 0, 1))
        state_frames = torch.as_tensor(state_frames, dtype=torch.float32).unsqueeze(0)
        state["image"] = state_frames.to(device)

        for t in range(T_MAX):
            if steps_done in [1000, 10000, 20000, 30000, 50000, 80000, 100000, 200000, 400000, 500000, 800000, 1200000]:
                print("steps: {}, time: {}, len memory: {}"
                      .format(steps_done, round(time.time()-init_time, 2), len(memory_collectsample)))
            # Update the current state
            curr_state = state.copy()

            # Select an action
            action = env.action_space.sample()

            # Interaction with the environment
            out_step = env.step(action)
            observation, reward, terminal, return_her, is_carrying \
                = out_step[0], out_step[1], out_step[2], out_step[3], out_step[4]

            observation["image"] = observation["image"][:, :, :2]
            observation_prep \
                = torch.as_tensor(observation['image'].transpose((2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)

            state_frames = torch.cat((curr_state["image"][:, 2:], observation_prep), dim=1)
            state["image"] = state_frames

            # Update the number of steps
            steps_done += 1

            # Add transition
            #if curr_state["image"][0, 6, 3, 6] != 1:
            #    print("error")
            memory_collectsample.store_data(curr_state["image"], curr_state["mission"], 0)

            if len(memory_collectsample) == memory_collectsample.memory_size:
                max_steps_reached = 1
                break
            # Terminate the episode if terminal state
            if terminal:
                if is_carrying:
                    if use_imc:
                        if reward == 0:
                            target_imc = torch.tensor([0], dtype=torch.long).to(device)
                            #memory_collectsample.list_of_targets += \
                            #    [0] * min(len(memory_collectsample.stored_data), dict_agent["n_keep_correspondence"])
                        else:
                            target_imc = torch.tensor([1], dtype=torch.long).to(device)
                            #memory_collectsample.list_of_targets += \
                            #    [1] * min(len(memory_collectsample.stored_data), dict_agent["n_keep_correspondence"])
                        memory_collectsample.add_stored_data(target_imc, dict_agent["n_keep_correspondence"])

                    elif use_her:
                        if reward == 0:
                            if return_her:
                                hindsight_reward = out_step[5]
                                hindsight_target = out_step[6]
                                target_mission = utils.mission_tokenizer(dict_env, hindsight_target).to(device)
                                memory_collectsample.add_stored_data(target_mission,
                                                                     dict_agent["n_keep_correspondence"])
                        else:
                            memory_collectsample.add_stored_data(state["mission"], dict_agent["n_keep_correspondence"])
                break

        if max_steps_reached:
            break
        env.close()
    print("num steps", steps_done)
    print("num ep", episode)
    return memory_collectsample


if __name__ == "__main__":

    dict_env = {
        "name": "5x5-C4-N2-O2",
        "game_type": "fetch",
        "size": 5,
        "numObjs": 2,
        "manual": 0,
        "random_target": 1,
        "oneobject": 0,
        "COLOR_TO_IDX": {
            "red": 0,
            "green": 1,
            "blue": 2,
            "purple": 3
        },
        "TYPE_TO_IDX": {
            "key": 0,
            "ball": 1
        },
        "T_max": 500,
    }

    dict_agent = {
        "name": "double-dqn-her",
        "agent": "double-dqn-her",
        "seed": 29,
        "frames": 4,
        "n_keep_correspondence": 1,
        "skew_ratio": 0.5,
        "memory_size": 1e4
    }

    if "her" in dict_agent["agent"]:
        use_her = 1
        use_imc = 0
    else:
        use_her = 0
        use_imc = 1
    mem = collect_samples(dict_env, dict_agent, use_her=use_her, use_imc=use_imc)

#    with open("collect_samples_{}_nc_{}_memory_size_{}_frames_number_of_steps.pkl".format(dict_agent["n_keep_correspondence"],
#                              int(dict_agent["memory_size"]), dict_agent["frames"]), 'wb') as output:
#        dill.dump(mem, output, dill.HIGHEST_PROTOCOL)