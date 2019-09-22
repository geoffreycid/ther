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
    "PAD": 0,
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
    "BEG": 25,
    "END": 26
}

idx2word = {
    0: "PAD",
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
    25: "BEG",
    26: "END"
}


# Load the memory
with open('/home/gcideron/datasets/collect_samples_1100_memory_size_4_frames_300_missions_her_cpu_rnn_shuffle_attrib.pkl', 'rb') as file:
    mem = dill.load(file)
print("Memory loaded")

config = {
    "log_every": 500,
    "earlystopping": 20,
    "iterations_before_earlystopping": 1e5,
    "n_iterations": 2e5,
    "batch_size": 128,
    "device": "cuda",
    "acc_min": 0.65
    }

device = "cuda"
importlib.reload(prednet)

net = prednet.PredMissionRNN(c=4, frames=1, n_words=27, word_embedding_size=128,
                             hidden_size=256, teacher_forcing_ratio=0.2, word2idx=word2idx,
                             idx2word=idx2word, lr=1e-4).to(device)

net.optimize_model(mem, config)
