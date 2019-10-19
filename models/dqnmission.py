import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAgent(nn.Module):
    def __init__(self):
        super(BaseAgent, self).__init__()
        self.mission = False

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.choice(range(self.n_actions))
        else:
            # max(1) for the dim, [1] for the indice, [0] for the value
            action = int(self.forward(state).max(1)[1].detach())

        return action


class DQN(BaseAgent):

    def __init__(self, h, w, c, n_actions, frames, num_token, device):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.mission = True
        self.embedded_dim = 16
        self.device = device
        self.num_token = num_token

        self.conv_net = nn.Sequential(
            nn.Conv2d(c * frames, 16, (2, 2)),
            nn.ReLU(),
            #nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        output_conv_h = h-3 #((h - 1) // 2 - 2)  # h-3 without maxpooling
        output_conv_w = w-3 #((w - 1) // 2 - 2)  # w-3 without maxpooling

        size_after_conv = 64 * output_conv_h * output_conv_w

        #self.fc = nn.Sequential(
        #    nn.Linear(in_features=(size_after_conv+64), out_features=64),
        #    nn.ReLU(),
        #    nn.Linear(in_features=64, out_features=n_actions)
        #)

        self.fcwo = nn.Sequential(
            nn.Linear(in_features=size_after_conv, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_actions)
        )

        #self.language_net = nn.Sequential(
        #    nn.Linear(in_features=self.num_token, out_features=self.embedded_dim),
        #    nn.ReLU(),
        #    nn.Linear(in_features=self.embedded_dim, out_features=64)
        #)

    def forward(self, state):
        out_conv = self.conv_net(state["image"])
        flatten = out_conv.view(out_conv.shape[0], -1)
        #out_language = self.language_net(state["mission"])
        #concat = torch.cat((flatten, out_language), dim=1)
        #return self.fc(concat)
        return self.fcwo(flatten)

    # Optimize the model
    def optimize_model(self, memory, target_net, optimizer, dict_agent):
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
        targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32, device=self.device)
        targets[batch_terminal == 0] = targets[batch_terminal == 0] \
                                       + dict_agent["gamma"] \
                                       * target_net(batch_next_state_non_terminal_dict).max(1)[0].detach()
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
        optimizer.zero_grad()
        loss.backward()
        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        # Do the gradient descent step
        optimizer.step()
