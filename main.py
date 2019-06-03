import configwrapper
import ray

if __name__ == '__main__':
    with open('configs/envs/fetch.json', 'r') as myfile:
        config_env = myfile.read()

    with open('configs/agents/fetch/doubledqnper.json', 'r') as myfile:
        config_agent = myfile.read()

    with open('configs/agents/fetch/gridsearch.json', 'r') as myfile:
        config_gridsearch = myfile.read()
    ray.shutdown()
    ray.init(
        temp_dir='/tmp/ray2',
        num_gpus=2,
        num_cpus=4,

             )
    configwrapper.wrapper(config_env=config_env, config_agent=config_agent, config_gridsearch=config_gridsearch)
    ray.shutdown()