import wrapper
import train
import ray
import json
import numpy as np

if __name__ == '__main__':

    @ray.remote(num_gpus=0.5, max_calls=1)
    def training(dict_env, dict_agent, dict_expert):
        return train.training(dict_env=dict_env, dict_agent=dict_agent, dict_expert=dict_expert)

    with open('configs/envs/fetch.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/duelingdoubledqn.json', 'r') as myfile:
        config_agent_simple = myfile.read()

    with open('configs/experts/expert_to_learn_rnn.json', 'r') as myfile:
        config_expert_to_learn = myfile.read()

    with open('configs/experts/expert_to_learn_rnn_scratch.json', 'r') as myfile:
        config_expert_to_learn_scratch = myfile.read()

    with open('configs/experts/her_expert.json', 'r') as myfile:
        config_her_expert = myfile.read()

    with open('configs/experts/no_expert.json', 'r') as myfile:
        config_no_expert = myfile.read()

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

    dict_her_expert = json.loads(config_her_expert)
    dict_no_expert = json.loads(config_no_expert)
    dict_expert_to_learn = json.loads(config_expert_to_learn)
    dict_expert_to_learn_scratch = json.loads(config_expert_to_learn_scratch)

    ray.init(
        temp_dir='/tmp/ray2',
    )

    dicts_expert = [dict_her_expert, dict_no_expert, dict_expert_to_learn, dict_expert_to_learn_scratch]

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
