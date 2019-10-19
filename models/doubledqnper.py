import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.doubledqn import DoubleDQN


class DoubleDQNPER(DoubleDQN):

    def __init__(self, h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(DoubleDQNPER, self).__init__(h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text)

    # Optimize the model
    def optimize_model(self, memory, target_net, dict_agent):
        # Wait for a min length before starting the optimization
        if len(memory) < dict_agent["batch_size"]:
            return

        # Sample from the memory replay
        transitions, is_weights, transition_idxs = memory.sample(dict_agent["batch_size"])

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
        # TD-Error
        td_errors = torch.abs(targets - predictions).detach().reshape(-1)
        memory.update(transition_idxs, td_errors.cpu().numpy())
        # Loss
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        criterion = F.smooth_l1_loss(predictions, targets, reduction='none') * is_weights
        loss = criterion.mean()
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        # Do the gradient descent step
        self.optimizer.step()

