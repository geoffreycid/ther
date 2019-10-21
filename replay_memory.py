import collections
import operator
import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, size, seed):
        self.transition = collections.namedtuple("Transition",
                                                 ["curr_state", "action", "reward", "next_state", "terminal",
                                                  "mission"])
        self.stored_transitions = []
        self.memory_size = int(size)
        self.memory = [None for _ in range(self.memory_size)]
        self.position = 0
        self.len = 0
        random.seed(seed)

    def add_transition(self, curr_state, action, reward, next_state, terminal, mission):
        self.memory[self.position] = self.transition(curr_state, action, reward, next_state, terminal, mission)
        # self.memory.append(self.transition(curr_state, action, reward, next_state, terminal, mission))
        # if len(self.memory) > self.memory_size:
        #    del self.memory[0]

        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0

    def sample(self, batch_size):
        if self.len == self.memory_size:
            return random.sample(self.memory, batch_size)
        else:
            return random.sample(self.memory[:self.len], batch_size)

    def __len__(self):
        return self.len

    def store_transition(self, curr_state, action, reward, next_state, terminal, mission):
        self.stored_transitions.append(self.transition(curr_state, action, reward, next_state, terminal, mission))

    def add_hindsight_transitions(self, reward, mission, keep_last_transitions):
        # keep_last_transitions = 0 => keep the whole episode
        if keep_last_transitions == 0:
            keep = 0
        elif keep_last_transitions > 0:
            keep = max(len(self.stored_transitions) - keep_last_transitions, 0)
        # Update the last transition with hindsight replay
        self.memory[self.position] = self.stored_transitions[-1]._replace(reward=reward, mission=mission)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0
        # Update all the transitions of the current episode with hindsight replay
        for transition in self.stored_transitions[keep:-1]:
            self.memory[self.position] = transition._replace(mission=mission)
            # Update the position and the len of the memory size
            self.position += 1
            self.len = min(self.memory_size, self.len + 1)
            if self.position > self.memory_size - 1:
                self.position = 0

    def erase_stored_transitions(self):
        self.stored_transitions = []

    def add_dense_transitions(self, reward, mission, action, keep_last_transitions_dense):
        # keep_last_transitions = 0 => keep the whole episode
        if keep_last_transitions_dense == 0:
            keep = 0
        elif keep_last_transitions_dense > 0:
            keep = max(len(self.stored_transitions) - keep_last_transitions_dense, 0)
        # Update the last transition with hindsight replay
        self.memory[self.position] = self.stored_transitions[-1]._replace(reward=reward, mission=mission,
                                                                          action=action, terminal=True)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0
        # Update all the transitions of the current episode with hindsight replay
        for transition in self.stored_transitions[keep:-1]:
            self.memory[self.position] = transition._replace(mission=mission)
            # Update the position and the len of the memory size
            self.position += 1
            self.len = min(self.memory_size, self.len + 1)
            if self.position > self.memory_size - 1:
                self.position = 0


class ReplayMemoryExpert(object):
    def __init__(self, size, seed):
        self.imc = collections.namedtuple("imc", ["state", "target"])
        self.memory_size = int(size)
        self.position = 0
        self.len = 0
        self.memory = [None for _ in range(self.memory_size)]
        self.episodes_done = 0
        self.memory_dense_size = self.memory_size * 10
        self.memory_dense = [None for _ in range(self.memory_dense_size)]
        self.position_dense = 0
        self.imc_dense = collections.namedtuple("dense", ["state", "target"])
        self.len_dense = 0
        self.stored_dense_data = []
        random.seed(seed)
        np.random.seed(seed)

    def add_data(self, curr_state, target):
        self.memory[self.position] = self.imc(curr_state, target)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_dense_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0

    def add_data_dense(self, curr_state, target):
        self.memory_dense[self.position_dense] = self.imc_dense(curr_state, target)
        self.position_dense += 1
        self.len_dense = min(self.memory_dense_size, self.len_dense + 1)
        if self.position_dense > self.memory_size * 10 - 1:
            self.position_dense = 0


class PrioritizedReplayMemory(object):
    def __init__(self, size, seed, alpha, beta, annealing_rate, eps=1e-6):
        self.transition = collections.namedtuple("Transition",
                                                 ["curr_state", "action", "reward", "next_state", "terminal",
                                                  "mission"])
        self.memory_size = int(size)
        self.memory = [None for _ in range(self.memory_size)]
        self.priorities = np.zeros(self.memory_size)

        self.position = 0
        # Parameters to modulate the amount of PER
        self.alpha = alpha
        self.beta = beta
        # Minimal probability
        self.eps = eps
        # Annealing beta
        self.annealing_rate = annealing_rate
        self.len = 0
        random.seed(seed)
        np.random.seed(seed)
        self.stored_transitions = []

    def add_transition(self, curr_state, action, reward, next_state, terminal, mission):
        self.memory[self.position] = \
            self.transition(curr_state=curr_state, action=action, reward=reward, next_state=next_state,
                            terminal=terminal, mission=mission)
        # Add the maximal priority
        if self.len == 0:
            self.priorities[self.position] = 1
        else:
            self.priorities[self.position] = self.priorities.max()

        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0

    def sample(self, batch_size):
        #normalized_priorities = np.power(self.priorities[:self.len], self.alpha) + self.eps
        #normalized_priorities /= normalized_priorities.sum()
        normalized_priorities = self.priorities[:self.len] / self.priorities[:self.len].sum()
        transition_idxs = np.random.choice(np.arange(self.len),
                                           size=batch_size, replace=False, p=normalized_priorities)
        self.beta = min(1, self.beta + self.annealing_rate)
        is_weights = np.power(self.len * self.priorities[transition_idxs], -self.beta)
        is_weights = is_weights / is_weights.max()
        op = operator.itemgetter(*transition_idxs)
        return op(self.memory), is_weights, transition_idxs

    def update(self, idxs, errors):
        errors = np.power(errors + self.eps, self.alpha)
        self.priorities[idxs] = errors

    def __len__(self):

        return self.len

    def store_transition(self, curr_state, action, reward, next_state, terminal, mission):
        self.stored_transitions.append(self.transition(curr_state, action, reward, next_state, terminal, mission))

    def add_hindsight_transitions(self, reward, mission, keep_last_transitions):
        # keep_last_transitions = 0 => keep the whole episode
        if keep_last_transitions == 0:
            keep = 0
        elif keep_last_transitions > 0:
            keep = max(len(self.stored_transitions) - keep_last_transitions, 0)
        # Update the last transition with hindsight replay
        self.memory[self.position] = self.stored_transitions[-1]._replace(reward=reward, mission=mission)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0
        # Update all the transitions of the current episode with hindsight replay
        for transition in self.stored_transitions[keep:-1]:
            self.memory[self.position] = transition._replace(mission=mission)
            # Update the position and the len of the memory size
            self.position += 1
            self.len = min(self.memory_size, self.len + 1)
            if self.position > self.memory_size - 1:
                self.position = 0

        self.erase_stored_transitions()

    def erase_stored_transitions(self):
        self.stored_transitions = []
