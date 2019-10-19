import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDoubleDQN(nn.Module):

    def __init__(self, h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        lr: learning rate
        num_token: number of words, useful only for the onehot modelisation
        device: device to use
        use_memory: boolean, 1: the frames are processed with a LSTM, 0: the frames are stacked to make a state
        """
        super(BaseDoubleDQN, self).__init__()

        self.n_actions = n_actions
        self.mission = True
        self.embedded_dim = 32
        self.device = device
        self.num_token = num_token
        self.use_memory = use_memory
        self.use_text = use_text
        self.frames = frames
        self.c = c
        self.h = h
        self.w = w

        output_conv_h = ((h - 1) // 2 - 2)  # h-3 without maxpooling
        output_conv_w = ((w - 1) // 2 - 2)  # w-3 without maxpooling

        self.size_after_conv = 64 * output_conv_h * output_conv_w

        if self.use_memory:
            self.memory_rnn = nn.LSTM(self.size_after_conv, self.size_after_conv, batch_first=True)

        frames_conv_net = 1 if use_memory else self.frames

        self.conv_net = nn.Sequential(
            nn.Conv2d(c * frames_conv_net, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(27, self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
            self.out = nn.Linear(in_features=self.text_embedding_size, out_features=self.embedded_dim)
        else:
            self.language_net = nn.Sequential(
                nn.Linear(in_features=self.num_token, out_features=self.embedded_dim)
            )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.choice(range(self.n_actions))
        else:
            # max(1) for the dim, [1] for the indice, [0] for the value
            if self.use_text:
                copy_state = state.copy()
                copy_state["text_length"] = [state["mission"].shape[0]]
                copy_state["mission"] = state["mission"].unsqueeze(0)
                action = int(self.forward(copy_state).max(1)[1].detach())
            else:
                action = int(self.forward(state).max(1)[1].detach())

        return action

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

        if self.use_text:
            text_length = [None] * dict_agent["batch_size"]
            for ind, mission in enumerate(batch_transitions.mission):
                text_length[ind] = mission.shape[0]
            batch_text_length = torch.tensor(text_length, dtype=torch.long).to(self.device)
            batch_mission = nn.utils.rnn.pad_sequence(batch_transitions.mission, batch_first=True).to(self.device)
        else:
            batch_mission = torch.cat(batch_transitions.mission)

        # Compute targets according to the Bellman eq
        if self.use_text:
            batch_next_state_non_terminal_dict = {
                "image": batch_next_state[batch_terminal == 0],
                "mission": batch_mission[batch_terminal == 0],
                "text_length": batch_text_length[batch_terminal == 0]
            }
        else:

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
        if self.use_text:
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
        predictions = self.forward(batch_curr_state_dict).gather(1, batch_action)

        # Loss
        loss = F.smooth_l1_loss(predictions, targets)
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        # Do the gradient descent step
        self.optimizer.step()
