import torch
import torch.nn as nn
from models.basedoubledqn import BaseDoubleDQN


class DoubleDQN(BaseDoubleDQN):

    def __init__(self, h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(DoubleDQN, self).__init__(h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text)

        # FC architecture after the concatenation of the two modalities
        self.fc = nn.Sequential(
            nn.Linear(in_features=(self.size_after_conv+self.embedded_dim), out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_actions)
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, state):

        if self.use_memory:
            batch_dim = state["image"].shape[0]
            out_conv = self.conv_net(state["image"].reshape(-1, self.c, self.h, self.w))
            out_conv = out_conv.reshape(batch_dim, self.frames, -1)
            hiddens = self.memory_rnn(out_conv)
            # hiddens = (outputs, (h_t, c_t))
            flatten = hiddens[1][0][0]
        else:
            out_conv = self.conv_net(state["image"])
            flatten = out_conv.view(out_conv.shape[0], -1)

        if self.use_text:
            # state["mission"] contains list of indices
            embedded = self.word_embedding(state["mission"])
            # Pack padded batch of sequences for RNN module
            packed = nn.utils.rnn.pack_padded_sequence(embedded, state["text_length"],
                                                       batch_first=True, enforce_sorted=False)
            # Forward pass through GRU
            outputs, hidden = self.text_rnn(packed)
            out_language = self.out(hidden[0])

        else:
            out_language = self.language_net(state["mission"])
        concat = torch.cat((flatten, out_language), dim=1)
        return self.fc(concat)
