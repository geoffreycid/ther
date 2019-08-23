# Import
import importlib
import dill
import random
import operator
import numpy as np
import torch
import models.predmissionnet as prednet
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

word2idx = {
    "pad": 0,
    "get": 1,
    "a": 2,
    "go": 3,
    "fetch": 4,
    "you": 5,
    "must": 6,
    "red": 7,
    "green": 8,
    "blue": 9,
    "purple": 10,
    "yellow": 11,
    "grey": 12,
    "key": 13,
    "ball": 14,
    "veryyoung": 15,
    "young": 16,
    "middle": 17,
    "old": 18,
    "veryold": 19,
    "verysmall": 20,
    "small": 21,
    "average": 22,
    "big": 23,
    "verybig": 24,
    "start": 25,
    ".": 26
}

idx2word = {
    0: "pad",
    1: "get",
    2: "a",
    3: "go",
    4: "fetch",
    5:"you",
    6: "must",
    7: "red",
    8: "green",
    9: "blue",
    10: "purple",
    11: "yellow",
    12: "grey",
    13: "key",
    14: "ball",
    15: "veryyoung",
    16: "young",
    17: "middle",
    18: "old",
    19: "veryold",
    20: "verysmall",
    21: "small",
    22: "average",
    23: "big",
    24: "verybig",
    25: "start",
    26: "."
}


# Load the memory
with open('/home/gcideron/datasets/collect_samples_11000_memory_size_4_frames_300_missions_her_cpu_rnn_shuffle_attrib.pkl', 'rb') as file:
    mem = dill.load(file)
print("Memory loaded")

random.shuffle(mem.memory)
train_memory = mem.memory[:1000]
test_memory = mem.memory[1000:]

miss_colors = ["red", "green", "blue", "purple", "yellow", "grey"]
miss_types = ["key", "ball"]
miss_sizes = ["verysmall", "small", "average", "big", "verybig"]
miss_seniorities = ["veryyoung", "young", "middle", "old", "veryold"]
all_missions = []
for color in miss_colors:
    for type in miss_types:
        for size in miss_sizes:
            for seniority in miss_seniorities:
                all_missions.append([word2idx[color], word2idx[type], word2idx[size], word2idx[seniority]])

random.shuffle(all_missions)
all_missions = torch.sort(torch.tensor(all_missions), dim=1)[0]
train_missions = all_missions[:int(all_missions.shape[0] * 0.9), :]
hold_out_missions = all_missions[int(all_missions.shape[0] * 0.9):, :]

memory_with_train_missions = []
memory_with_hold_out_missions = []
for imc in train_memory:
    if 4 in torch.eq(torch.sort(imc.target[-5:-1])[0], train_missions).sum(dim=1):
        memory_with_train_missions.append(imc)
    else:
        memory_with_hold_out_missions.append(imc)

test_memory_with_train_missions = memory_with_train_missions[-1000:]
memory_with_train_missions = memory_with_train_missions[:-1000]

device = "cuda"
importlib.reload(prednet)

#net = prednet.PredMissionNet(h=7, w=7, c=4, frames=1, lr_imc=5e-5,
#                            dim_tokenizer=18, weight_decay=0).to(device)

#net = prednet.PredMissionOneHot(c=4, frames=1, n_type=num_types, n_color=num_colors,
#                                n_seniority=num_seniority, n_size=num_size, lr=1e-5).to(device)

#net = prednet.PredMissionOneHotDense(c=4, frames=1, n_type=num_types, n_color=num_colors,
#                                n_seniority=num_seniority, n_size=num_size, lr=1e-5).to(device)
#net = prednet.PredMissionMultiLabel(c=4, frames=1, n_type=num_types, n_color=num_colors,
#                                n_seniority=num_seniority, n_size=num_size, lr=1e-5).to(device)

net = prednet.PredMissionRNN(c=4, frames=1, n_words=27, word_embedding_size=128, hidden_size=512,
                             teacher_forcing_ratio=0.5, lr=1e-4).to(device)

config = {
    "log_every": 1000,
    "earlystopping": 20,
    "iterations_before_earlystopping": 3e5,
    "n_iterations": 2e6,
    "batch_size": 128,
    "device": "cuda",
    "acc_min": 0.65
}

# Config
n_iterations = config["n_iterations"]
batch_size = config["batch_size"]

# Early stopping parameters
earlystopping = config["earlystopping"]
iterations_before_early_stopping = config["iterations_before_earlystopping"]

# Optimization steps
steps_done = 0

test_accs = np.array([])

for j in range(int(n_iterations)):
    #imcs = random.sample(train_memory, batch_size)
    imcs = random.sample(memory_with_train_missions[:1000], batch_size)

    batch_imcs = mem.imc(*zip(*imcs))
    batch_states = torch.cat(batch_imcs.state)[:, 12:].to(config["device"])

    # Pad all the sequences, dim = (seq_len, batch)
    batch_targets = nn.utils.rnn.pad_sequence(batch_imcs.target).to(config["device"])

    net.optimizer.zero_grad()

    loss = net.forward(batch_states, batch_targets)
    loss.backward()
    # Keep the gradient between (-10, 10). Works like one uses L1 loss for large gradients (see Huber loss)
    for param in net.parameters():
        param.grad.data.clamp_(-10, 10)
    net.optimizer.step()

    steps_done += 1

    if steps_done % config["log_every"] == 0:

        print("Iteration : ", j+1)
        print("loss", loss.item())
        # Accuracy on train missions but different images
        imcs = test_memory_with_train_missions
        batch_imcs = mem.imc(*zip(*imcs))
        batch_states = torch.cat(batch_imcs.state)[:, 12:].to(config["device"])
        pred_idxs = net.prediction(batch_states)

        acc_whole_mission = 0
        accuracy = 0
        for ind, mission in enumerate(batch_imcs.target):
            count = 0
            for idx in mission[-5:-1]:
                accuracy += idx in pred_idxs[ind]
                count += idx in pred_idxs[ind]
            if count == 4:
                acc_whole_mission += 1
        accuracy /= len(batch_imcs.target) * 4
        acc_whole_mission /= len(batch_imcs.target)

        print("Accuracy on train missions {}".format(round(accuracy, 3)))
        print("Whole mission accuracy on train missions {}".format(round(acc_whole_mission, 3)))

        # Accuracy on hold out missions
        imcs = memory_with_hold_out_missions
        batch_imcs = mem.imc(*zip(*imcs))
        batch_states = torch.cat(batch_imcs.state)[:, 12:].to(config["device"])
        pred_idxs = net.prediction(batch_states)

        acc_whole_mission = 0
        accuracy = 0
        for ind, mission in enumerate(batch_imcs.target):
            count = 0
            for idx in mission[-5:-1]:
                accuracy += idx in pred_idxs[ind]
                count += idx in pred_idxs[ind]
            if count == 4:
                acc_whole_mission += 1
        accuracy /= len(batch_imcs.target) * 4
        acc_whole_mission /= len(batch_imcs.target)

        print("Accuracy on hold out missions {}".format(round(accuracy, 3)))
        print("Whole mission accuracy on hold out missions {}".format(round(acc_whole_mission, 3)))

        pred_sentences = []
        for i in range(pred_idxs.size(0)):
            idxs = pred_idxs[i]
            sentence = ''
            for idx in idxs:
                sentence += ' ' + idx2word[idx.item()]
                if idx == word2idx["."]:
                    break
            pred_sentences.append(sentence)
        print("predictions of sentences \n", pred_sentences[:5])

        true_idxs = nn.utils.rnn.pad_sequence(batch_imcs.target).to(config["device"]).t()
        true_sentences = []
        for i in range(true_idxs.size(0)):
            idxs = true_idxs[i]
            sentence = ''
            for idx in idxs:
                sentence += ' ' + idx2word[idx.item()]
            true_sentences.append(sentence)
        print("true sentences \n", true_sentences[:5])
        print(" ")


