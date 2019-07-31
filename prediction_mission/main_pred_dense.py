import random
import dill
import ray
import collections
import train_pred

if __name__ == '__main__':
    @ray.remote(num_gpus=0.5, max_calls=1)
    def training(train_memory, train_memory_dense, test_memory, test_memory_dense, config):
        return train_pred.optimization(train_memory=train_memory, train_memory_dense=train_memory_dense,
                                       test_memory=test_memory, test_memory_dense=test_memory_dense, config=config)

    # Load the memory
    with open('/home/gcideron/home/gcideron/visual_her/collect_samples_110000_memory_size_4_frames_300_missions_her_cpu_dense_pickup.pkl', 'rb') as file:
        mem_dense = dill.load(file)
    print("Memory loaded")

    train_memory_onehot = mem_dense.memory[:100000]
    test_memory_onehot = mem_dense.memory[100000:]

    train_memory_dense = mem_dense.memory_dense[:540000]
    test_memory_dense = mem_dense.memory_dense[540000:]

    with open('config.json', 'r') as myfile:
        config_json = myfile.read()
    config = json.loads(config_json)
    # config dense
    config_dense = config.copy()
    config_dense["imc"] = 0
    config_dense["dense"] = 1
    config_dense["ratio_dense_onehot"] = 0.1

    list_of_train = [mem_dense.memory[:1000], mem_dense.memory[:5000], mem_dense.memory[:8000],
                            mem_dense.memory[:10000], mem_dense.memory[:20000],
                            mem_dense.memory[:50000], mem_dense.memory[:80000], mem_dense.memory[:100000]]

    list_of_train_dense = [mem_dense.memory_dense[:1000*5], mem_dense.memory_dense[:5000*5],
                           mem_dense.memory_dense[:8000*5], mem_dense.memory_dense[:10000*5],
                           mem_dense.memory_dense[:20000*5], mem_dense.memory_dense[:50000*5],
                           mem_dense.memory_dense[:80000*5], mem_dense.memory_dense[:100000*5]]

    # Use Ray to do the allocation of resources
    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=3,
        num_cpus=6,
    )
    ray.get([training.remote(net, train_memory, train_memory_dense, test_memory_onehot, test_memory_dense, config_dense)
             for (train_memory, train_memory_dense) in list(zip(list_of_train, list_of_train_dense))])

    ray.shutdown()
