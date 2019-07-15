import torch
import torch.nn as nn
import torch.nn.functional as F

class PredMissionNet(nn.Module):

    def __init__(self, h, w, c, frames, lr_imc, dim_tokenizer, weight_decay, device):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """

        super(PredMissionNet, self).__init__()

        self.embedded_dim = 16
        self.device = device
        self.dim_tokenizer = dim_tokenizer

        output_conv_h = ((h - 1) // 2 - 2)  # h-3 without maxpooling
        output_conv_w = ((w - 1) // 2 - 2)  # w-3 without maxpooling

        self.embedding_image_conv = nn.Sequential(
            nn.Conv2d(c * frames, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        #self.embedding_image_as_fc = nn.Sequential(
        #    nn.Linear(in_features=98, out_features=32),
        #    nn.ReLU(),
        #    nn.Linear(in_features=32, out_features=64),
        #    nn.ReLU(),
        #    nn.Linear(in_features=64, out_features=128),
        #)
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

        self.params_imc = list(self.embedding_image_conv.parameters()) \
                          + list(self.embedding_image_fc.parameters()) \
                          + list(self.embedding_mission_fc.parameters()) \
                          + list(self.tiny_fc.parameters())

        self.optimizer_imc = torch.optim.Adam(self.params_imc, lr=lr_imc, weight_decay=weight_decay)

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

    def find_best_mission(self, state, missions):
        # Note: state [1, x] and missions [n, y]
        embedded_state = self.embedding_image(state.repeat(missions.shape[0], 1, 1, 1)).detach()
        embedded_missions = self.embedding_mission(missions).detach()
        distances = F.pairwise_distance(embedded_state, embedded_missions)
        indices_best_missions = torch.argsort(distances, descending=False)
        return missions[indices_best_missions[0]]

    def pred_correspondence(self, state, mission, target):
        """
        :param state: dim: [1, x]
        :param mission: dim: [1, y]
        :return:
        """
        pred = torch.argmax(self.image_mission_correspondence(state, mission), dim=1)
        return target == pred

    def optimize_imc(self, memory_imc, dict_agent):
        if len(memory_imc) < dict_agent["batch_size_imc"]:
            return

        imcs = memory_imc.sample(dict_agent["batch_size_imc"])

        batch_imcs = memory_imc.imc(*zip(*imcs))
        batch_state = torch.cat(batch_imcs.state)
        batch_mission = torch.cat(batch_imcs.mission)
        batch_target = torch.cat(batch_imcs.target)

        batch_predictions = self.image_mission_correspondence(batch_state, batch_mission)

        loss = self.criterion(batch_predictions, batch_target)
        self.optimizer_imc.zero_grad()
        loss.backward()
        self.optimizer_imc.step()


class PredMissionOneHot(nn.Module):
    def __init__(self, c, frames, n_type, n_color, lr):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(PredMissionOneHot, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(c * frames, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.type_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_type)
        )

        self.color_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_color)
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state):
        out_conv = self.conv_net(state)
        flatten = out_conv.view(out_conv.shape[0], -1)

        return self.type_fc(flatten), self.color_fc(flatten)

    def prediction(self, state):
        with torch.no_grad():
            pred_type, pred_color = self.forward(state)
            return pred_type.max(1)[1], pred_color.max(1)[1]
