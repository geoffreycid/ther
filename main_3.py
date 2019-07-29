import configwrapper
import train
import ray
import json
import aggregator.aggregator as aggregator


if __name__ == '__main__':

    @ray.remote(num_gpus=0.33, max_calls=1)
    def training(dict_env, dict_agent):
        return train.training(dict_env=dict_env, dict_agent=dict_agent)

    with open('configs/envs/fetch_3.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
        config_agent_simple = myfile.read()
    with open('configs/agents/fetch/doubledqnper.json', 'r') as myfile:
        config_agent_per = myfile.read()
    with open('configs/agents/fetch/doubledqnher.json', 'r') as myfile:
        config_agent_her = myfile.read()
    #with open('configs/agents/fetch/doubledqnimc.json', 'r') as myfile:
    #    config_agent_imc = myfile.read()

    with open('configs/agents/fetch/gridsearch.json', 'r') as myfile:
        config_gridsearch = myfile.read()
    with open('configs/agents/fetch/extension_3.json', 'r') as myfile:
        config_extension = myfile.read()

    # Decode json files into dict
    dict_fetch = json.loads(config_env)

    grid_search = json.loads(config_gridsearch)
    extension = json.loads(config_extension)

    dict_agent_simple = json.loads(config_agent_simple)
    #dict_agent_per = json.loads(config_agent_per)
    dict_agent_her = json.loads(config_agent_her)
    #dict_agent_imc = json.loads(config_agent_imc)

    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=1,
        num_cpus=2,
    )

    dict_simple, dicts_simple = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                      grid_search=grid_search, extension=extension)

    #dict_per, dicts_per = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_per,
    #                                            grid_search=grid_search, extension=extension)

    dict_her, dicts_her = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_her,
                                                grid_search=grid_search, extension=extension)

    #dict_imc, dicts_imc = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_imc,
    #                                            grid_search=grid_search, extension=extension)

    #dicts_to_train = dicts_her + dicts_per + dicts_simple
    dicts_to_train = dicts_her + dicts_simple

    # Use Ray to do the allocation of resources
    ray.get([training.remote(dict_env=dict_fetch, dict_agent=d) for d in dicts_to_train])
    ray.shutdown()

    # Aggregate all seeds
    aggregator.wrapper(dict_simple["agent_dir"], output="summary")
    #aggregator.wrapper(dict_per["agent_dir"], output="summary")
    aggregator.wrapper(dict_her["agent_dir"], output="summary")
    #aggregator.wrapper(dict_imc["agent_dir"], output="summary")