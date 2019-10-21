import random
import torch.nn as nn


class DQNCartpole(nn.Module):

    def __init__(self, n_actions):
        """
        h: height of the screen
        w: width of the screen
        frames: number of frames taken into account for the state
        n_actions: number of actions
        """
        super(DQNCartpole, self).__init__()

        self.n_actions = n_actions

        self.fc = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=n_actions)
        )

    def forward(self, x):
        flatten = x.view(x.shape[0], -1)
        return self.fc(flatten)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            action = random.choice(range(self.n_actions))
        else:
            # max(1) for the dim, [1] for the indice, [0] for the value
            action = int(self.forward(state).max(1)[1].detach())
            # Update the number of steps done

        return action
