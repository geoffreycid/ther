import random
import dill
import ray
import collections
import json
import train_pred
import models.predmissionnet as prednet


if __name__ == '__main__':
    @ray.remote(num_gpus=0.5, max_calls=1)
    def training(net, train_memory, test_memory, config):
        return train_pred.optimization(net=net, train_memory=train_memory, test_memory=test_memory, config=config)

    # Load the memory
    with open('collect_samples_110000_memory_size_4_frames_300_missions_her_cpu.pkl', 'rb') as file:
        mem_onehot = dill.load(file)
    with open('collect_samples_110000_memory_size_4_frames_300_missions_imc_cpu.pkl', 'rb') as file:
        mem_imc = dill.load(file)
    print("Memory loaded")

    random.shuffle(mem_imc.memory)
    train_memory_imc = mem_imc.memory[:100000]
    test_memory_imc = mem_imc.memory[100000:]

    random.shuffle(mem_onehot.memory)
    train_memory_onehot = mem_onehot.memory[:100000]
    test_memory_onehot = mem_onehot.memory[100000:]

    config_imc = {
        "imc": 1,
        "n_iterations": 5e5,
        "save_every": 100,
        "batch_size": 128,
        "seed": 32,
        "frames": 4,
        "channels": 4,
        "num_types": 2,
        "num_colors": 6,
        "num_seniority": 5,
        "num_size": 5,
        "tuple_imc": collections.namedtuple("imc", ["state", "mission", "target"]),
        "dir": "/home/gcideron/home/gcideron/visual_her/logs_pred",
        "earlystopping": 1000,
        "iterations_before_earlystopping": 10000,
    }

    list_of_train_imc = [mem_imc.memory[:1000], mem_imc.memory[:5000], mem_imc.memory[:8000],
                     mem_imc.memory[:10000], mem_imc.memory[:20000],
                         mem_imc.memory[:50000], mem_imc.memory[:80000], mem_imc.memory[:100000]]

    net_imc = prednet.PredMissionNet(h=7, w=7, c=4, frames=1, lr_imc=5e-5,
                                dim_tokenizer=18, weight_decay=0).to("cuda")



    # One hot
    config_onehot = config_imc.copy()
    config_onehot["imc"] = 0

    net_onehot = prednet.PredMissionOneHot(c=4, frames=1, n_type=config_onehot["num_types"],
                                           n_color=config_onehot["num_colors"],
                                           n_seniority=config_onehot["num_seniority"],
                                           n_size=config_onehot["num_size"], lr=1e-5).to("cuda")

    list_of_train_onehot = [mem_onehot.memory[:1000], mem_onehot.memory[:5000], mem_onehot.memory[:8000],
                            mem_onehot.memory[:10000], mem_onehot.memory[:20000],
                            mem_onehot.memory[:50000], mem_onehot.memory[:80000], mem_onehot.memory[:100000]]

    list_of_train = list_of_train_imc + list_of_train_onehot

    len_list = 8
    list_of_nets = [net_imc] * len_list + [net_onehot] * len_list
    list_of_test = [test_memory_imc] * len_list + [test_memory_onehot] * len_list
    list_of_config = [config_imc] * len_list + [config_onehot] * len_list

    # Use Ray to do the allocation of resources
    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=4,
        num_cpus=8,
    )
    #ray.get([training.remote(net, train_memory, test_memory_imc, config_imc) for train_memory in list_of_train_imc])
    ray.get([training.remote(net, train_memory, test_memory, config)
             for (net, train_memory, test_memory, config)
             in list(zip(list_of_nets, list_of_train, list_of_test, list_of_config))])
    ray.shutdown()
