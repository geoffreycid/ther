import dill
import ray
import collections
import json
import train_pred


if __name__ == '__main__':
    @ray.remote(num_gpus=0.33, max_calls=1)
    def training(train_memory, test_memory, config, train_memory_dense, test_memory_dense):
        return train_pred.optimization(train_memory=train_memory, test_memory=test_memory, config=config,
                                       train_memory_dense=train_memory_dense, test_memory_dense=test_memory_dense)

    # Load memories
    with open('/home/gcideron/visual_her/datasets/collect_samples_110000_memory_size_4_frames_300_missions_imc_cpu.pkl', 'rb') as file:
        mem_imc = dill.load(file)
    with open('/home/gcideron/visual_her/datasets/collect_samples_110000_memory_size_4_frames_300_missions_her_cpu_dense_pickup.pkl', 'rb') as file:
        mem_onehot_dense = dill.load(file)
    print("Memories loaded")

    # len max
    len_max_train = 100000
    len_max_train_dense = len_max_train * 5

    # Configs for imc, onehot, and dense
    with open('config.json', 'r') as myfile:
        config_json = myfile.read()

    config = json.loads(config_json)
    config["tuple_imc"] = collections.namedtuple("imc", ["state", "mission", "target"])
    config["tuple_imc_dense"] = collections.namedtuple("dense", ["state", "target"])

    # config imc
    config_imc = config.copy()
    config_imc["imc"] = 1
    config_imc["dense"] = 0
    config_imc["lr"] = 5e-5
    config_imc["weight_decay"] = 0

    # config onehot
    config_onehot = config.copy()
    config_onehot["imc"] = 0
    config_onehot["dense"] = 0

    # config dense
    config_dense = config.copy()
    config_dense["imc"] = 0
    config_dense["dense"] = 1
    config_dense["ratio_dense_onehot"] = 0.1

    # test sets
    test_memory_imc = mem_imc.memory[len_max_train:]
    test_memory_onehot = mem_onehot_dense.memory[len_max_train:]
    test_memory_dense = mem_onehot_dense.memory_dense[len_max_train_dense:]

    # train sets
    list_of_train_imc = [mem_imc.memory[:200], mem_imc.memory[:500], mem_imc.memory[:1000],
                         mem_imc.memory[:5000], mem_imc.memory[:10000],
                         mem_imc.memory[:50000], mem_imc.memory[:80000], mem_imc.memory[:100000]]

    list_of_train_onehot = [mem_onehot_dense.memory[:200], mem_onehot_dense.memory[:500],
                            mem_onehot_dense.memory[:1000], mem_onehot_dense.memory[:5000],
                            mem_onehot_dense.memory[:10000], mem_onehot_dense.memory[:50000],
                            mem_onehot_dense.memory[:80000], mem_onehot_dense.memory[:100000]]

    list_of_train = list_of_train_onehot + list_of_train_onehot + list_of_train_imc

    len_list = len(list_of_train_imc)

    # list of test sets
    list_of_test = [test_memory_onehot] * len_list + [test_memory_onehot] * len_list + [test_memory_imc] * len_list

    # list of test dense
    list_of_test_dense = [None] * len_list + [test_memory_dense] * len_list + [None] * len_list

    # list of train dense
    list_of_train_dense = [mem_onehot_dense.memory_dense[:1000], mem_onehot_dense.memory_dense[:2500],
                            mem_onehot_dense.memory_dense[:5000], mem_onehot_dense.memory_dense[:25000],
                            mem_onehot_dense.memory_dense[:50000], mem_onehot_dense.memory_dense[:250000],
                            mem_onehot_dense.memory_dense[:400000], mem_onehot_dense.memory_dense[:500000]]

    list_of_train_dense = [None] * len_list + list_of_train_dense + [None] * len_list

    # list of configs
    list_of_config = [config_onehot] * len_list + [config_dense] * len_list + [config_imc] * len_list

    # Use Ray to do the allocation of resources
    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=2,
        num_cpus=6,
    )

    ray.get([training.remote(train_memory, test_memory, config, train_memory_dense, test_memory_dense)
             for (train_memory, test_memory, config, train_memory_dense, test_memory_dense)
             in list(zip(list_of_train, list_of_test, list_of_config, list_of_train_dense, list_of_test_dense))])
    ray.shutdown()
