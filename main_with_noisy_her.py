import wrapper
import train
import ray
import json
from copy import deepcopy
import numpy as np
#import aggregator.aggregator as aggregator


if __name__ == '__main__':

    @ray.remote(num_gpus=0.33, max_calls=1)
    def training(dict_env, dict_agent, dict_expert):
        return train.training(dict_env=dict_env, dict_agent=dict_agent, dict_expert=dict_expert)

    with open('configs/envs/fetch_noisy_her.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/duelingdoubledqn.json', 'r') as myfile:
        config_agent_simple = myfile.read()

    with open('configs/experts/her_expert.json', 'r') as myfile:
        config_her_expert = myfile.read()

    with open('configs/experts/no_expert.json', 'r') as myfile:
        config_no_expert = myfile.read()

    with open('configs/experts/noisy_her_noise_random_0_1.json', 'r') as myfile:
        config_her_noise_random_0_1 = myfile.read()

    dict_her = json.loads(config_her_expert)
    dict_no_expert = json.loads(config_no_expert)
    dict_her_noise_0_1 = json.loads(config_her_noise_random_0_1)

    dicts_expert = {}
    for proba in [0.2, 0.4, 0.6, 0.8]:
        dicts_expert["proba_{}".format(proba)] = {}
        dicts_expert["proba_{}".format(proba)] = deepcopy(dict_her_noise_0_1)
        dicts_expert["proba_{}".format(proba)]["name"] = "noisy-her-noise-random-{}".format(proba)
        dicts_expert["proba_{}".format(proba)]["parameters-noisy-her"]["proba-noisy"] = proba

    dicts_expert = [dict_her] + [dict_no_expert] + list(dicts_expert.values())
    #dicts_expert = list(dicts_expert.values())

    # Grid search + extension
    with open('configs/agents/fetch/gridsearch.json', 'r') as myfile:
        config_gridsearch = myfile.read()
    with open('configs/agents/fetch/extension.json', 'r') as myfile:
        config_extension = myfile.read()


    # Decode json files into dict
    dict_fetch = json.loads(config_env)

    grid_search = json.loads(config_gridsearch)
    extension = json.loads(config_extension)

    dict_agent_simple = json.loads(config_agent_simple)

    ray.init(
        temp_dir='/tmp/ray2'
    )

    dicts_to_train = []
    for dict_expert in dicts_expert:
        dicts_to_train += wrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                          dict_expert=dict_expert, grid_search=grid_search,
                                          extension=extension)[1]

    dicts_expert = list(np.repeat(dicts_expert, len(extension["seed"])))

    # Use Ray to do the allocation of resources
    ray.get([training.remote(dict_env=dict_fetch, dict_agent=agent, dict_expert=expert)
             for (agent, expert) in list(zip(dicts_to_train, dicts_expert))])
    ray.shutdown()

    # Aggregate all seeds
    #aggregator.wrapper(dict_simple["agent_dir"], output="summary")
    #aggregator.wrapper(dict_per["agent_dir"], output="summary")
    #aggregator.wrapper(dict_her["agent_dir"], output="summary")
    #aggregator.wrapper(dict_imc["agent_dir"], output="summary")