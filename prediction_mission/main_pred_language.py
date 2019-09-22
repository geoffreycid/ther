import dill
import ray
import collections
import sys
sys.path.extend(['/home/gcideron/visual_her', '/home/gcideron/visual_her'])
import aggregator.aggregator as aggregator
import random
import torch
import json
import train_pred_language as train_pred


def training_with_one_dataset(mem, run_id):
    @ray.remote(num_gpus=0.33, max_calls=1)
    def training(train_memory, test_memory, test_memory_with_hold_out_missions, config):
        return train_pred.optimization(train_memory=train_memory, test_memory=test_memory,
                                                   test_memory_with_hold_out_missions=test_memory_with_hold_out_missions,
                                                   config=config)

    # Load the memory
    #with open(
    #        '/home/gcideron/datasets/collect_samples_11000_memory_size_4_frames_300_missions_her_cpu_rnn_shuffle_attrib.pkl',
    #        'rb') as file:
    #    mem = dill.load(file)
    #print("Memory loaded")

    # Configs for imc, onehot, and dense
    with open('config_language.json', 'r') as myfile:
        config_json = myfile.read()

    config = json.loads(config_json)
    config["dir_logs"] = config["dir_logs"] + "/run_{}".format(run_id)

    # Datasets
    word2idx = config["word2idx"]
    miss_colors = ["red", "green", "blue", "purple", "yellow", "grey"]
    miss_types = ["key", "ball"]
    miss_sizes = ["tiny", "small", "medium", "large", "giant"]
    miss_shades = ["very_light", "light", "neutral", "dark", "very_dark"]
    all_missions = []
    for color in miss_colors:
        for type in miss_types:
            for size in miss_sizes:
                for shade in miss_shades:
                    all_missions.append([word2idx[color], word2idx[type], word2idx[size], word2idx[shade]])

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

    test_memory = memory_with_train_missions[-5000:]
    train_memory = memory_with_train_missions[:-5000]
    test_memory_with_hold_out_missions = test_memory_with_hold_out_missions[:5000]

    print(len(test_memory))
    print(len(test_memory_with_hold_out_missions))
    print(len(train_memory))

    # train sets

    list_of_train = [train_memory[:200], train_memory[:500],
                            train_memory[:750], train_memory[:1000],
                            train_memory[:1500], train_memory[:2000],
                            train_memory[:5000], train_memory[:10000],
                            train_memory[:15000], train_memory[:20000]]

    list_of_train = list_of_train

    len_list = len(list_of_train)

    # list of test sets
    list_of_test = [test_memory] * len_list
    list_of_test_with_hold_out_missions = [test_memory_with_hold_out_missions] * len_list

    # list of configs
    list_of_config = [config] * len_list

    # Use Ray to do the allocation of resources
    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=2,
        num_cpus=6,
    )

    ray.get([training.remote(train_memory, test_memory, test_memory_with_hold_out_missions, config)
             for (train_memory, test_memory, test_memory_with_hold_out_missions, config)
             in list(zip(list_of_train, list_of_test, list_of_test_with_hold_out_missions, list_of_config))])
    ray.shutdown()


if __name__ == '__main__':
    # Load the memory
    with open('/home/gcideron/datasets/memory_size_60000_seed_1.pkl', 'rb') as file:
        mem_1 = dill.load(file)
    with open('/home/gcideron/datasets/memory_size_60000_seed_2.pkl', 'rb') as file:
        mem_2 = dill.load(file)
    with open('/home/gcideron/datasets/memory_size_60000_seed_3.pkl', 'rb') as file:
        mem_3 = dill.load(file)
    with open('/home/gcideron/datasets/memory_size_60000_seed_4.pkl', 'rb') as file:
        mem_4 = dill.load(file)
    with open('/home/gcideron/datasets/memory_size_60000_seed_5.pkl', 'rb') as file:
        mem_5 = dill.load(file)
    print('loaded memories')

    training_with_one_dataset(mem_1, 1)
    training_with_one_dataset(mem_2, 2)
    training_with_one_dataset(mem_3, 3)
    training_with_one_dataset(mem_4, 4)
    training_with_one_dataset(mem_5, 5)

    # Aggregates
    with open('config_language.json', 'r') as myfile:
        config_json = myfile.read()

    config = json.loads(config_json)
    
    aggregator.wrapper(config["dir_logs"], output="summary")
