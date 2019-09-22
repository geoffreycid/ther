import dill
import collections
import random
import torch
import json
import train_pred_language as train_pred

# Load the memory
with open(
        '/home/gcideron/datasets/collect_samples_11000_memory_size_4_frames_300_missions_her_cpu_rnn_shuffle_attrib.pkl',
        'rb') as file:
    mem = dill.load(file)
print("Memory loaded")

# len max
len_max_train = 100000

# Configs for imc, onehot, and dense
with open('config_language.json', 'r') as myfile:
    config_json = myfile.read()

config = json.loads(config_json)

# Datasets
word2idx = config["word2idx"]
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
train_missions = all_missions[:int(all_missions.shape[0] * 0.8), :]
hold_out_missions = all_missions[int(all_missions.shape[0] * 0.8):, :]

memory_with_train_missions = []
test_memory_with_hold_out_missions = []
for imc in mem.memory:
    if 4 in torch.eq(torch.sort(imc.target[-4:])[0], train_missions).sum(dim=1):
        memory_with_train_missions.append(imc)
    else:
        test_memory_with_hold_out_missions.append(imc)

test_memory = memory_with_train_missions[-1000:]
train_memory = memory_with_train_missions[:-1000]

print(len(test_memory))
print(len(test_memory_with_hold_out_missions))
print(len(train_memory))


train_pred.optimization(train_memory=train_memory, test_memory=test_memory,
                                               test_memory_with_hold_out_missions=test_memory_with_hold_out_missions,
                                               config=config)