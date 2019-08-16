import configwrapper
import train
import ray
import json
#import aggregator.aggregator as aggregator


if __name__ == '__main__':

    @ray.remote(num_gpus=0.5, max_calls=1)
    def training(dict_env, dict_agent, dict_expert):
        return train.training(dict_env=dict_env, dict_agent=dict_agent, dict_expert=dict_expert)

    with open('configs/envs/fetch.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/doubledqn.json', 'r') as myfile:
        config_agent_simple = myfile.read()
#    with open('configs/agents/fetch/doubledqnper.json', 'r') as myfile:
#        config_agent_per = myfile.read()

    with open('configs/experts/expert_to_learn.json', 'r') as myfile:
        config_expert_to_learn = myfile.read()
    with open('configs/experts/expert_to_learn_dense.json', 'r') as myfile:
        config_expert_to_learn_dense = myfile.read()

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
#    dict_agent_per = json.loads(config_agent_per)

    dict_her_expert = json.loads(config_her_expert)
    dict_no_expert = json.loads(config_no_expert)
    dict_expert_to_learn = json.loads(config_expert_to_learn)
    dict_expert_to_learn_dense = json.loads(config_expert_to_learn_dense)

    ray.init(
        temp_dir='/tmp/ray2',
    )

    dict_dqn_no_expert, dicts_dqn_no_expert = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                      dict_expert=dict_no_expert, grid_search=grid_search,
                                                      extension=extension)

    # dict_per_no_expert, dicts_per_no_expert = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_per,
    #                                            dict_expert=dict_no_expert, grid_search=grid_search,
    #                                            extension=extension)

    dict_dqn_her, dicts_dqn_her = configwrapper.wrapper(dict_env=dict_fetch, dict_agent=dict_agent_simple,
                                                dict_expert=dict_her_expert, grid_search=grid_search,
                                                extension=extension)

    dict_dqn_expert_to_learn, dicts_dqn_expert_to_learn = configwrapper.wrapper(dict_env=dict_fetch,
                                                                                    dict_agent=dict_agent_simple,
                                                                                    dict_expert=dict_expert_to_learn,
                                                                                    grid_search=grid_search,
                                                                                    extension=extension)

    dict_dqn_expert_to_learn_dense, dicts_dqn_expert_to_learn_dense = configwrapper.wrapper(dict_env=dict_fetch,
                                                                                    dict_agent=dict_agent_simple,
                                                                                    dict_expert=dict_expert_to_learn_dense,
                                                                                    grid_search=grid_search,
                                                                                    extension=extension)

    dicts_to_train = dicts_dqn_expert_to_learn_dense + dicts_dqn_her + dicts_dqn_no_expert + dicts_dqn_expert_to_learn

    nb_seed = len(dicts_dqn_no_expert)
    #dicts_expert = [dict_expert_to_learn_dense] * nb_seed + [dict_expert_to_learn] * nb_seed + [
    #    dict_her_expert] * nb_seed + [dict_no_expert] * nb_seed

    dicts_expert = [dict_expert_to_learn_dense] * nb_seed + [dict_her_expert] * nb_seed + [dict_no_expert] * nb_seed \
                    + [dict_expert_to_learn] * nb_seed

    # Use Ray to do the allocation of resources
    ray.get([training.remote(dict_env=dict_fetch, dict_agent=agent, dict_expert=expert)
             for (agent, expert) in list(zip(dicts_to_train, dicts_expert))])
    ray.shutdown()

    # Aggregate all seeds
    #aggregator.wrapper(dict_simple["agent_dir"], output="summary")
    #aggregator.wrapper(dict_per["agent_dir"], output="summary")
    #aggregator.wrapper(dict_her["agent_dir"], output="summary")
    #aggregator.wrapper(dict_imc["agent_dir"], output="summary")