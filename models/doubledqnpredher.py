import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDQNPredHER(nn.Module):

    def __init__(self, h, w, c, n_actions, frames, lr, dict_env, num_token, device):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(DoubleDQNPredHER, self).__init__()

        self.n_actions = n_actions
        self.mission = True
        self.embedded_dim = 16
        self.device = device
        self.num_token = num_token

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
            nn.Linear(in_features=self.num_token, out_features=self.embedded_dim)
        #    nn.ReLU(),
        #    nn.Linear(in_features=self.embedded_dim, out_features=64)
        )

        self.prediction_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.prediction_fc = nn.Sequential(
            nn.Linear(in_features=size_after_conv, out_features=64),
            nn.ReLU()
        )
        self.prediction_type = nn.Sequential(
            nn.Linear(in_features=64, out_features=len(dict_env["COLOR_TO_IDX"].keys()))
        )
        self.prediction_color = nn.Sequential(
            nn.Linear(in_features=64, out_features=len(dict_env["TYPE_TO_IDX"].keys()))
        )

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.params_agent = list(self.conv_net_1.parameters()) + list(self.conv_net_2.parameters()) \
                        + list(self.language_net.parameters()) + list(self.fc.parameters())
        self.optimizer_agent = torch.optim.RMSprop(self.params_agent, lr=lr)

        self.params_pred_type = list(self.conv_net_1.parameters()) + list(self.prediction_conv.parameters()) \
                              + list(self.prediction_fc.parameters()) + list(self.prediction_type.parameters())

        self.params_pred_color = list(self.conv_net_1.parameters()) + list(self.prediction_conv.parameters()) \
                              + list(self.prediction_fc.parameters()) + list(self.prediction_color.parameters())

        self.optimizer_pred_mission = torch.optim.RMSprop(self.params_pred_type + self.params_pred_color, lr=lr)
        #self.optimizer_pred_color = torch.optim.RMSprop(params_pred_color, lr=lr)

    def forward_pred_mission(self, state):
        out = self.prediction_conv(self.conv_net_1(state["image"]))
        flatten = out.view(out.shape[0], -1)
        logits_type = self.prediction_type(self.prediction_fc(flatten))
        logits_color = self.prediction_color(self.prediction_fc(flatten))
        return logits_type, logits_color

    def pred_proba_mission(self, state):
        out = self.prediction_conv(self.conv_net_1(state["image"]))
        flatten = out.view(out.shape[0], -1)

        logits_type = self.prediction_type(self.prediction_fc(flatten))
        proba_type = F.softmax(logits_type, dim=0).detach()

        logits_color = self.prediction_color(self.prediction_fc(flatten))
        proba_color = F.softmax(logits_color, dim=0).detach()

        return proba_type, proba_color

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

    def optimize_mission_pred(self, state, target):
        logits_type, logits_color = self.forward_pred_mission(state)

        loss = self.criterion(logits_type, target["type"]) + self.criterion(logits_type, target["color"])
        #loss_color = self.criterion(logits_type, target["color"])

        self.optimizer_pred_mission.zero_grad()
        #self.optimizer_pred_color.zero_grad()

        loss.backward()
        #loss_color.backward()

        self.optimizer_pred_mission.step()
        #self.optimizer_pred_color.step()

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


