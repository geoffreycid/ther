import configwrapper
import train
import ray
import json
import aggregator.aggregator as aggregator


if __name__ == '__main__':

    @ray.remote(num_gpus=0.33, max_calls=1)
    def training(dict_env, dict_agent):
        return train.training(dict_env=dict_env, dict_agent=dict_agent)

    with open('configs/envs/fetch_1.json', 'r') as myfile:
        config_env_1 = myfile.read()
    with open('configs/envs/fetch_2.json', 'r') as myfile:
        config_env_2 = myfile.read()
    with open('configs/envs/fetch_3.json', 'r') as myfile:
        config_env_3 = myfile.read()

    with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
        config_agent_simple_1 = myfile.read()
    with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
        config_agent_simple_2 = myfile.read()
    with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
        config_agent_simple_3 = myfile.read()

    with open('configs/agents/fetch/doubledqnper.json', 'r') as myfile:
        config_agent_per = myfile.read()

    with open('configs/agents/fetch/doubledqnher.json', 'r') as myfile:
        config_agent_her_1 = myfile.read()
    with open('configs/agents/fetch/doubledqnher.json', 'r') as myfile:
        config_agent_her_2 = myfile.read()
    with open('configs/agents/fetch/doubledqnher.json', 'r') as myfile:
        config_agent_her_3 = myfile.read()
    #with open('configs/agents/fetch/doubledqnimc.json', 'r') as myfile:
    #    config_agent_imc = myfile.read()

    with open('configs/agents/fetch/gridsearch.json', 'r') as myfile:
        config_gridsearch = myfile.read()

    with open('configs/agents/fetch/extension_1.json', 'r') as myfile:
        config_extension_1 = myfile.read()
    with open('configs/agents/fetch/extension_2.json', 'r') as myfile:
        config_extension_2 = myfile.read()
    with open('configs/agents/fetch/extension_3.json', 'r') as myfile:
        config_extension_3 = myfile.read()

    # Decode json files into dict
    dict_fetch_1 = json.loads(config_env_1)
    dict_fetch_2 = json.loads(config_env_2)
    dict_fetch_3 = json.loads(config_env_3)

    grid_search = json.loads(config_gridsearch)
    extension_1 = json.loads(config_extension_1)
    extension_2 = json.loads(config_extension_2)
    extension_3 = json.loads(config_extension_3)

    dict_agent_simple_1 = json.loads(config_agent_simple_1)
    dict_agent_simple_2 = json.loads(config_agent_simple_2)
    dict_agent_simple_3 = json.loads(config_agent_simple_3)

    #dict_agent_per = json.loads(config_agent_per)

    dict_agent_her_1 = json.loads(config_agent_her_1)
    dict_agent_her_2 = json.loads(config_agent_her_2)
    dict_agent_her_3 = json.loads(config_agent_her_3)

    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=2,
        num_cpus=6,
    )

    dict_simple_1, dicts_simple_1 = configwrapper.wrapper(dict_env=dict_fetch_1, dict_agent=dict_agent_simple_1,
                                                      grid_search=grid_search, extension=extension_1)

    dict_simple_2, dicts_simple_2 = configwrapper.wrapper(dict_env=dict_fetch_2, dict_agent=dict_agent_simple_2,
                                                      grid_search=grid_search, extension=extension_2)

    dict_simple_3, dicts_simple_3 = configwrapper.wrapper(dict_env=dict_fetch_3, dict_agent=dict_agent_simple_3,
                                                      grid_search=grid_search, extension=extension_3)

    #dict_per, dicts_per = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_per,
    #                                            grid_search=grid_search, extension=extension)

    dict_her_1, dicts_her_1 = configwrapper.wrapper(dict_env=dict_fetch_1, dict_agent=dict_agent_her_1,
                                                grid_search=grid_search, extension=extension_1)
    dict_her_2, dicts_her_2 = configwrapper.wrapper(dict_env=dict_fetch_2, dict_agent=dict_agent_her_2,
                                                grid_search=grid_search, extension=extension_2)
    dict_her_3, dicts_her_3 = configwrapper.wrapper(dict_env=dict_fetch_3, dict_agent=dict_agent_her_3,
                                                grid_search=grid_search, extension=extension_3)

    #dict_imc, dicts_imc = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_imc,
    #                                            grid_search=grid_search, extension=extension)

    #dicts_to_train = dicts_her + dicts_per + dicts_simple
    dicts_to_train = dicts_her_1 + dicts_simple_1 + dicts_her_2 + dicts_simple_2 + dicts_her_3 + dicts_simple_3

    dicts_fect = [dict_fetch_1, dict_fetch_1, dict_fetch_2, dict_fetch_2, dict_fetch_3, dict_fetch_3]

    # Use Ray to do the allocation of resources
    ray.get([training.remote(dict_env=dict_fect, dict_agent=d) for (dict_fect, d) in list(zip(dicts_fect, dicts_to_train))])
    ray.shutdown()

    # Aggregate all seeds
    #aggregator.wrapper(dict_simple["agent_dir"], output="summary")
    #aggregator.wrapper(dict_per["agent_dir"], output="summary")
    #aggregator.wrapper(dict_her["agent_dir"], output="summary")
    #aggregator.wrapper(dict_imc["agent_dir"], output="summary")