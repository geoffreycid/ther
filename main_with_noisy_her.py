import configwrapper
import train
import ray
import json
#import aggregator.aggregator as aggregator


if __name__ == '__main__':

    @ray.remote(num_gpus=0.33, max_calls=1)
    def training(dict_env, dict_agent, dict_expert):
        return train.training(dict_env=dict_env, dict_agent=dict_agent, dict_expert=dict_expert)

    with open('configs/envs/fetch.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
        config_agent_simple = myfile.read()

    with open('configs/experts/her_expert.json', 'r') as myfile:
        config_her_expert = myfile.read()
    with open('configs/experts/noisy_her_mask_color_0_5.json', 'r') as myfile:
        config_her_mask_color_0_5 = myfile.read()
    with open('configs/experts/noisy_her_mask_color_0_2.json', 'r') as myfile:
        config_her_mask_color_0_2 = myfile.read()
    with open('configs/experts/noisy_her_mask_random_0_5.json', 'r') as myfile:
        config_her_mask_random_0_5 = myfile.read()
    with open('configs/experts/noisy_her_noise_random_0_5.json', 'r') as myfile:
        config_her_noise_random_0_5 = myfile.read()
    with open('configs/experts/noisy_her_noise_random_0_2.json', 'r') as myfile:
        config_her_noise_random_0_2 = myfile.read()

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

    dict_her = json.loads(config_her_expert)
    dict_her_mask_color_0_5 = json.loads(config_her_mask_color_0_5)
    dict_her_mask_color_0_2 = json.loads(config_her_mask_color_0_2)
    dict_her_mask_random_0_5 = json.loads(config_her_mask_random_0_5)
    dict_her_noise_random_0_5 = json.loads(config_her_noise_random_0_5)
    dict_her_noise_random_0_2 = json.loads(config_her_noise_random_0_2)


    ray.init(
        temp_dir='/tmp/ray2'
    )

    dict_dqn_her, dicts_dqn_her = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her, grid_search=grid_search,
                                                extension=extension)

    dict_dqn_her_mask_color_0_5, dicts_dqn_her_mask_color_0_5 = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her_mask_color_0_5, grid_search=grid_search,
                                                extension=extension)

    dict_dqn_her_mask_color_0_2, dicts_dqn_her_mask_color_0_2 = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her_mask_color_0_2, grid_search=grid_search,
                                                extension=extension)

    dict_dqn_her_mask_random_0_5, dicts_dqn_her_mask_random_0_5 = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her_mask_random_0_5, grid_search=grid_search,
                                                extension=extension)

    dict_dqn_her_noise_random_0_5, dicts_dqn_her_noise_random_0_5 = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her_noise_random_0_5, grid_search=grid_search,
                                                extension=extension)
    dict_dqn_her_noise_random_0_2, dicts_dqn_her_noise_random_0_2 = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her_noise_random_0_2, grid_search=grid_search,
                                                extension=extension)

    dicts_to_train = dicts_dqn_her + dicts_dqn_her_mask_color_0_5 + dicts_dqn_her_mask_color_0_2 \
                     + dicts_dqn_her_mask_random_0_5 + dicts_dqn_her_noise_random_0_5 + dicts_dqn_her_noise_random_0_2

    dicts_expert = [dict_her, dict_her_mask_color_0_5, dict_her_mask_color_0_2,
                    dict_her_mask_random_0_5, dict_her_noise_random_0_5, dict_her_noise_random_0_2]

    # Use Ray to do the allocation of resources
    ray.get([training.remote(dict_env=dict_fetch, dict_agent=agent, dict_expert=expert)
             for (agent, expert) in list(zip(dicts_to_train, dicts_expert))])
    ray.shutdown()

    # Aggregate all seeds
    #aggregator.wrapper(dict_simple["agent_dir"], output="summary")
    #aggregator.wrapper(dict_per["agent_dir"], output="summary")
    #aggregator.wrapper(dict_her["agent_dir"], output="summary")
    #aggregator.wrapper(dict_imc["agent_dir"], output="summary")