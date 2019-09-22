import configwrapper
import train_continual_learning as train
import ray
import json
import numpy as np
from copy import deepcopy
#import aggregator.aggregator as aggregator


if __name__ == '__main__':

    @ray.remote(num_gpus=0.5, max_calls=1)
    def training(dict_env, dict_agent, dict_expert):
        return train.training(dict_env=dict_env, dict_agent=dict_agent, dict_expert=dict_expert)

    with open('configs/envs/fetch_hold_out_missions.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/duelingdoubledqn_continual.json', 'r') as myfile:
        config_agent = myfile.read()

    with open('configs/experts/learned_expert_continual.json', 'r') as myfile:
        config_expert = myfile.read()

    # Grid search + extension
    with open('configs/agents/fetch/gridsearch.json', 'r') as myfile:
        config_gridsearch = myfile.read()
    with open('configs/agents/fetch/extension.json', 'r') as myfile:
        config_extension = myfile.read()


    # Decode json files into dict
    dict_fetch = json.loads(config_env)
    grid_search = json.loads(config_gridsearch)
    extension = json.loads(config_extension)
    dict_agent = json.loads(config_agent)
    dict_expert = json.loads(config_expert)

    #dicts_agent = []
    #for i in range(5):
    #    d = deepcopy(dict_agent)
    #    d["weights_path"] = dict_agent["weights_path"] + "agent_model_steps_3000000_run_{}.pt".format(i)
    #    dicts_agent.append(d)


    #dicts_expert = []
    #for i in range(5):
    #    d = deepcopy(dict_expert)
    #    d["weights_path"] = dict_agent["weights_path"] + "expert_model_steps_3000000_run_{}.pt".format(i)
    #    dicts_expert.append(d)


    ray.init(
        temp_dir='/tmp/ray2',
    )

    dicts_to_train = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent,
                                            dict_expert=dict_expert, grid_search=grid_search,
                                            extension=extension)[1]

    dicts_expert = list(np.repeat(dict_expert, len(extension["seed"])))

    # Use Ray to do the allocation of resources
    ray.get([training.remote(dict_env=dict_fetch, dict_agent=agent, dict_expert=expert)
             for (agent, expert) in list(zip(dicts_to_train, dicts_expert))])
    ray.shutdown()

    # Aggregate all seeds
    #aggregator.wrapper(dict_simple["agent_dir"], output="summary")
    #aggregator.wrapper(dict_per["agent_dir"], output="summary")
    #aggregator.wrapper(dict_her["agent_dir"], output="summary")
    #aggregator.wrapper(dict_imc["agent_dir"], output="summary")