import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import operator
from sklearn.metrics import f1_score


class PredMissionImc(nn.Module):

    def __init__(self, h, w, c, frames, lr_imc, dim_tokenizer, weight_decay):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """

        super(PredMissionImc, self).__init__()

        self.embedded_dim = 32
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
    def __init__(self, c, frames, n_type, n_color, n_seniority, n_size, lr):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """
        super(PredMissionOneHot, self).__init__()

        self.n_type = n_type
        self.n_color = n_color
        self.n_seniority = n_seniority
        self.n_size = n_size

        self.use_dense = 0

        self.conv_net = nn.Sequential(
            nn.Conv2d(c * frames, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.type_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=n_type)
        )

        self.color_fc = nn.Sequential(
            #nn.Linear(in_features=64, out_features=64),
            #nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_color)
        )

        self.seniority_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=n_seniority)
        )

        self.size_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=n_size)
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state):
        out_conv = self.conv_net(state)
        flatten = out_conv.view(out_conv.shape[0], -1)
        shared = self.shared_fc(flatten)

        return self.type_fc(shared), self.color_fc(shared), self.seniority_fc(shared), self.size_fc(shared)

    def prediction(self, state):
        with torch.no_grad():
            pred_type, pred_color, pred_seniority, pred_size = self.forward(state)
            return pred_type.max(1)[1], pred_color.max(1)[1], pred_seniority.max(1)[1], pred_size.max(1)[1]

    def prediction_mission(self, state):
        pred_type, pred_color, pred_seniority, pred_size = self.prediction(state)

        type_onehot = torch.eye(self.n_type)[pred_type]
        color_onehot = torch.eye(self.n_color)[pred_color]
        seniority_onehot = torch.eye(self.n_seniority)[pred_seniority]
        size_onehot = torch.eye(self.n_size)[pred_size]

        mission = (type_onehot, color_onehot, seniority_onehot, size_onehot)
        return torch.cat(mission, dim=1)

    def optimize_model(self, memory, config):

        # Create the train and the test set
        len_memory = memory.len
        len_memory_train = int(len_memory * 0.95)
        len_test = len_memory - len_memory_train
        train_memory = memory.memory[:len_memory_train]
        test_memory = memory.memory[len_memory_train:len_memory]

        # Config
        n_iterations = config["n_iterations"]
        batch_size = config["batch_size"]

        if len_memory < batch_size:
            return 0

        # Choose the device
        if "device" in config:
            device = config["device"]
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Early stopping parameters
        earlystopping = config["earlystopping"]
        iterations_before_early_stopping = config["iterations_before_earlystopping"]

        # Optimization steps
        steps_done = 0

        test_accs = np.array([])

        while True:

            imcs = random.sample(train_memory, batch_size)

            batch_imcs = memory.imc(*zip(*imcs))
            batch_states = torch.cat(batch_imcs.state)

            # Divide the target for each attribute
            batch_targets = torch.cat(batch_imcs.target)
            batch_type_targets = batch_targets[:, :self.n_type]
            batch_type_targets = batch_type_targets.argmax(1)

            batch_color_targets = batch_targets[:, self.n_type: self.n_type + self.n_color]
            batch_color_targets = batch_color_targets.argmax(1)

            batch_seniority_targets = batch_targets[:,
                                      self.n_type + self.n_color:self.n_type + self.n_color + self.n_seniority]
            batch_seniority_targets = batch_seniority_targets.argmax(1)

            batch_size_targets = batch_targets[:, self.n_type + self.n_color + self.n_seniority:]
            batch_size_targets = batch_size_targets.argmax(1)

            # Compute the predictions
            batch_type_predictions, batch_color_predictions, batch_seniority_predictions, batch_size_predictions \
                = self(batch_states)

            self.optimizer.zero_grad()

            type_loss = self.criterion(batch_type_predictions, batch_type_targets)
            color_loss = self.criterion(batch_color_predictions, batch_color_targets)
            seniority_loss = self.criterion(batch_seniority_predictions, batch_seniority_targets)
            size_loss = self.criterion(batch_size_predictions, batch_size_targets)

            loss = sum([type_loss, color_loss, seniority_loss, size_loss])
            loss.backward()
            self.optimizer.step()

            steps_done += 1

            if steps_done % config["log_every"] == 0:

                batch_imcs = memory.imc(*zip(*test_memory))
                batch_states = torch.cat(batch_imcs.state)

                # Predictions
                batch_type_predictions, batch_color_predictions, batch_seniority_predictions, batch_size_predictions \
                    = self.prediction(batch_states)
                batch_type_predictions_onehot = torch.eye(self.n_type)[batch_type_predictions]
                batch_color_predictions_onehot = torch.eye(self.n_color)[batch_color_predictions]
                batch_seniority_predictions_onehot = torch.eye(self.n_seniority)[batch_seniority_predictions]
                batch_size_predictions_onehot = torch.eye(self.n_size)[batch_size_predictions]

                miss = (batch_type_predictions_onehot, batch_color_predictions_onehot,
                        batch_seniority_predictions_onehot, batch_size_predictions_onehot)

                batch_mission_predictions = torch.cat(miss, dim=1).to(device)

                # Targets
                batch_targets = torch.cat(batch_imcs.target)

                # Compute accuracies
                acc_total = float(torch.all(torch.eq(batch_mission_predictions, batch_targets), dim=1).sum()) / len_test

                test_accs = np.append(test_accs, acc_total)
                # Early stopping
                if steps_done > iterations_before_early_stopping and test_accs.size > earlystopping - 1:
                    if np.sum(test_accs[-earlystopping] < test_accs[-earlystopping:]) == 0:
                        print("Early stopping with accuracy {}".format(acc_total))
                        break

            if steps_done > n_iterations:
                break
        print("accuracies", test_accs[-5:])
        # return test_accs[-1] > config["acc_min"], test_accs[-1]
        return 1, test_accs[-1]


class PredMissionOneHotDense(PredMissionOneHot):

    def __init__(self, c, frames, n_type, n_color, n_seniority, n_size, lr):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        """

        super(PredMissionOneHotDense, self).__init__(c, frames, n_type, n_color, n_seniority, n_size, lr)

        self.use_dense = 1

        self.dense_conv = nn.Sequential(
            nn.Conv2d(c * frames, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.dense_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2)
        )

        params_dense = list(self.dense_conv.parameters()) + list(self.dense_fc.parameters())

        self.optimizer_dense = torch.optim.RMSprop(params_dense, lr=lr)

    def forward_dense(self, state):
        out_conv = self.dense_conv(state)
        flatten = out_conv.view(out_conv.shape[0], -1)
        return self.dense_fc(flatten)

    def prediction_dense(self, state):
        pred_dense = self.forward_dense(state)
        return pred_dense.max(1)[1]

    def optimize_model_dense(self, memory, config):

        len_memory_dense = memory.len_dense
        len_memory_train_dense = int(len_memory_dense * 0.9)
        # len_test_dense = len_memory_dense - len_memory_train_dense

        train_memory_dense = memory.memory_dense[:len_memory_train_dense]
        test_memory_dense = memory.memory_dense[len_memory_train_dense:len_memory_dense]

        # Config
        n_iterations = config["n_iterations"]
        batch_size = config["batch_size"]

        if len_memory_dense < batch_size:
            return 0
        # Choose the device
        if "device" in config:
            device = config["device"]
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Early stopping parameters
        earlystopping = config["earlystopping"]
        iterations_before_early_stopping = config["iterations_before_earlystopping"]

        # Optimization steps
        steps_done = 0

        train_targets_dense = torch.cat(memory.imc_dense(*zip(*train_memory_dense)).target).cpu().numpy()
        idx_targets_corres_dense = np.argwhere(train_targets_dense == 1).reshape(-1)
        idx_targets_no_corres_dense = np.argwhere(train_targets_dense == 0).reshape(-1)

        test_f1s = np.array([])

        while True:

            batch_size_corres_dense = batch_size // 2
            idx_corres_dense = np.random.choice(idx_targets_corres_dense, size=batch_size_corres_dense)
            idx_no_corres_dense = np.random.choice(idx_targets_no_corres_dense,
                                                   size=batch_size - batch_size_corres_dense)
            op_corres = operator.itemgetter(*idx_corres_dense)
            op_no_corres = operator.itemgetter(*idx_no_corres_dense)
            imcs_corres_dense = op_corres(train_memory_dense)
            imcs_no_corres_dense = op_no_corres(train_memory_dense)
            imcs_dense = imcs_corres_dense + imcs_no_corres_dense

            batch_imcs_dense = memory.imc_dense(*zip(*imcs_dense))
            batch_states_dense = torch.cat(batch_imcs_dense.state)
            batch_dense_targets = torch.cat(batch_imcs_dense.target)
            batch_dense_predictions = self.forward_dense(batch_states_dense)

            self.optimizer_dense.zero_grad()

            loss = self.criterion(batch_dense_predictions, batch_dense_targets)

            loss.backward()
            self.optimizer_dense.step()

            steps_done += 1

            if steps_done % config["log_every"] == 0:

                batch_imcs_dense = memory.imc_dense(*zip(*test_memory_dense))
                batch_states_dense = torch.cat(batch_imcs_dense.state)
                batch_dense_predictions = self.prediction_dense(batch_states_dense).to(device)
                batch_dense_targets = torch.cat(batch_imcs_dense.target)
                # acc_dense = float(torch.eq(batch_dense_predictions, batch_dense_targets).sum()) / len_test_dense
                f1 = f1_score(batch_dense_targets.cpu().numpy(), batch_dense_predictions.cpu().numpy())

                test_f1s = np.append(test_f1s, f1)
                # Early stopping
                if steps_done > iterations_before_early_stopping and test_f1s.size > earlystopping - 1:
                    if np.sum(test_f1s[-earlystopping] < test_f1s[-earlystopping:]) == 0:
                        print("Early stopping with f1 score {}".format(f1))
                        break

            if steps_done > n_iterations:
                break
        print("test f1s", test_f1s[-5:])
        # return test_f1s[-1] > config["f1_min"], test_f1s[-1]
        return 1, test_f1s[-1]