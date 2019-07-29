import random
from copy import deepcopy as copy
import operator
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDQNIMC(nn.Module):

    def __init__(self, h, w, c, n_actions, frames, lr, lr_imc, dim_tokenizer, device):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(DoubleDQNIMC, self).__init__()

        self.n_actions = n_actions
        self.mission = True
        self.embedded_dim = 32
        self.device = device
        self.dim_tokenizer = dim_tokenizer

        self.conv_net_1 = nn.Sequential(
            nn.Conv2d(c * frames, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2))
        )

        self.conv_net_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        output_conv_h = ((h - 1) // 2 - 2)  # h-3 without maxpooling
        output_conv_w = ((w - 1) // 2 - 2)  # w-3 without maxpooling

        size_after_conv = 64 * output_conv_h * output_conv_w

        self.fc = nn.Sequential(
            nn.Linear(in_features=(size_after_conv+self.embedded_dim), out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_actions)
        )

        self.language_net = nn.Sequential(
            nn.Linear(in_features=self.dim_tokenizer, out_features=self.embedded_dim)
        )

        self.embedding_image_conv = nn.Sequential(
            nn.Conv2d(c * 1, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.embedding_image_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=128)
        )

        self.embedding_mission_fc = nn.Sequential(
            nn.Linear(in_features=dim_tokenizer, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128)
        )

        self.tiny_fc = nn.Linear(in_features=1, out_features=2)

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.params_agent = list(self.conv_net_1.parameters()) + list(self.conv_net_2.parameters()) \
                        + list(self.language_net.parameters()) + list(self.fc.parameters())
        self.optimizer_agent = torch.optim.RMSprop(self.params_agent, lr=lr)

        #self.params_imc = list(self.conv_net_1.parameters()) + list(self.language_net.parameters()) \
        #                  + list(self.embedding_image_conv.parameters()) \
        #                  + list(self.embedding_image_fc.parameters()) + list(self.embedding_mission_fc.parameters()) \
        #                  + list(self.tiny_fc.parameters())
        self.params_imc = list(self.embedding_image_conv.parameters()) \
                          + list(self.embedding_image_fc.parameters()) \
                          + list(self.embedding_mission_fc.parameters()) \
                          + list(self.tiny_fc.parameters())
        self.optimizer_imc = torch.optim.Adam(self.params_imc, lr=lr_imc, weight_decay=1e-6)

    def embedding_image(self, batch_state):
        out_conv = self.embedding_image_conv(batch_state)
        flatten = out_conv.view(out_conv.shape[0], -1)
        out_fc = self.embedding_image_fc(flatten)
        #reshaped = batch_state.view(batch_state.shape[0], -1)
        #out_fc = self.embedding_image_as_fc(reshaped)
        embedded_image = F.normalize(out_fc, p=2, dim=1)
        return embedded_image

    def embedding_mission(self, batch_mission):
        out = self.embedding_mission_fc(batch_mission)
        return F.normalize(out, p=2, dim=1)

    def correspondence(self, embedding_image, embedding_mission):
        dists = F.pairwise_distance(embedding_image, embedding_mission, p=2, keepdim=True)
        return F.softmax(self.tiny_fc(dists), dim=1)

    def image_mission_correspondence(self, batch_state, batch_mission):
        embedded_images = self.embedding_image(batch_state)
        embedded_missions = self.embedding_mission(batch_mission)
        return self.correspondence(embedded_images, embedded_missions)

    def find_best_mission_one_state(self, state, missions):
        # Note: state [1, x] and missions [n, y]
        embedded_state = self.embedding_image(state.repeat(missions.shape[0], 1, 1, 1)).detach()
        embedded_missions = self.embedding_mission(missions).detach()
        distances = F.pairwise_distance(embedded_state, embedded_missions)
        indices_best_missions = torch.argsort(distances, descending=False)
        return missions[indices_best_missions[0]]

    def find_best_mission(self, states, missions):
        # Calculate the opposite of the dot product which has the same order as
        # the L2 distance but is can be computed only with a matrix product
        # Note: state [m, x] and missions [n, y]

        with torch.no_grad():
            embedded_states = self.embedding_image(states)
            embedded_missions = self.embedding_mission(missions)
        similarities = torch.mm(embedded_states, embedded_missions.t())
        indices_best_missions = similarities.argmax(dim=1)
        return torch.stack([missions[ind] for ind in indices_best_missions])

    def pred_correspondence(self, state, mission, target):
        """
        :param state: dim: [1, x]
        :param mission: dim: [1, y]
        :return:
        """
        pred = torch.argmax(self.image_mission_correspondence(state, mission), dim=1)
        return target == pred

    def forward(self, state):
        out_conv = self.conv_net_2(self.conv_net_1(state["image"]))
        flatten = out_conv.view(out_conv.shape[0], -1)
        out_language = self.language_net(state["mission"])
        concat = torch.cat((flatten, out_language), dim=1)
        return self.fc(concat)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.choice(range(self.n_actions))
        else:
            # max(1) for the dim, [1] for the indice, [0] for the value
            action = int(self.forward(state).max(1)[1].detach())

        return action

    def optimize_imc(self, memory_imc, dict_agent, all_possible_missions):
        len_memory = len(memory_imc)

        if len_memory < dict_agent["batch_size_imc"] or len_memory < dict_agent["min_len_imc"]:
            return

        memory = copy(memory_imc.memory)
        random.shuffle(memory)
        train_memory = memory[:int(0.9*len_memory)]
        test_memory = memory[int(0.9*len_memory):]

        # Subset of test with only good correspondence
        batch_imcs_test = memory_imc.imc(*zip(*test_memory))
        batch_targets_test = torch.cat(batch_imcs_test.target).cpu().numpy()
        inds_good_correspondence = np.argwhere(batch_targets_test == 1).reshape(-1)
        op = operator.itemgetter(*inds_good_correspondence)
        test_memory_good_correspondence = op(test_memory)

        batch_imcs_test_good_correspondence = memory_imc.imc(*zip(*test_memory_good_correspondence))
        batch_states_test_good_correspondence = torch.cat(batch_imcs_test_good_correspondence.state)[:, 9:]
        batch_missions_test_good_correspondence = torch.cat(batch_imcs_test_good_correspondence.mission)

        for _ in range(dict_agent["n_epochs_imc"]):
            beg_ind = 0
            end_ind = dict_agent["batch_size_imc"]
            for i in range(len_memory//dict_agent["batch_size_imc"]):
                imcs = train_memory[beg_ind:end_ind]
                batch_imcs = memory_imc.imc(*zip(*imcs))
                # Keep only the last frame
                batch_states = torch.cat(batch_imcs.state)[:, 9:]
                batch_missions = torch.cat(batch_imcs.mission)
                batch_predictions = self.image_mission_correspondence(batch_states, batch_missions)

                batch_targets = torch.cat(batch_imcs.target)

                loss = self.criterion(batch_predictions, batch_targets)
                self.optimizer_imc.zero_grad()
                loss.backward()
                self.optimizer_imc.step()
            # Test the net
            batch_predictions_test = self.find_best_mission(batch_states_test_good_correspondence,
                                                            all_possible_missions)
            accuracy = torch.all(torch.eq(batch_predictions_test, batch_missions_test_good_correspondence), dim=1)
            accuracy = accuracy.float().mean()
            print("accuracy", accuracy)
            if accuracy > 0.95:
                break

    # Optimize the model
    def optimize_model(self, memory, target_net, dict_agent):
        if len(memory) < dict_agent["batch_size"]:
            return

        # Sample from the memory replay
        transitions = memory.sample(dict_agent["batch_size"])
        # Batch the transitions into one namedtuple
        batch_transitions = memory.transition(*zip(*transitions))
        batch_curr_state = torch.cat(batch_transitions.curr_state)
        batch_next_state = torch.cat(batch_transitions.next_state)
        batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype=torch.int32)
        batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long, device=self.device).reshape(-1, 1)
        batch_mission = torch.cat(batch_transitions.mission)

        # Compute targets according to the Bellman eq
        batch_next_state_non_terminal_dict = {
            "image": batch_next_state[batch_terminal == 0],
            "mission": batch_mission[batch_terminal == 0]
        }

        # Evaluation of the Q value with the target net
        targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32, device=self.device).reshape(-1, 1)
        # Double DQN
        if torch.sum(batch_terminal) != dict_agent["batch_size"]:
            # Selection of the action with the policy net
            args_actions = self.forward(batch_next_state_non_terminal_dict).max(1)[1].reshape(-1, 1)
            targets[batch_terminal == 0] = targets[batch_terminal == 0] \
                                       + dict_agent["gamma"] \
                                       * target_net(batch_next_state_non_terminal_dict).gather(1, args_actions).detach()

        targets = targets.reshape(-1, 1)
        # Compute the current estimate of Q
        batch_curr_state_dict = {
            "image": batch_curr_state,
            "mission": batch_mission
        }
        predictions = self.forward(batch_curr_state_dict).gather(1, batch_action)

        # Loss
        loss = F.smooth_l1_loss(predictions, targets)
        # Optimization
        self.optimizer_agent.zero_grad()
        loss.backward()
        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for param in self.params_agent:
            param.grad.data.clamp_(-1, 1)
        # Do the gradient descent step
        self.optimizer_agent.step()


