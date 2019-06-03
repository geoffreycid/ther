# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:15:23 2019

@author: geoffreycideron
"""
import os
import sys
import json
import time
from itertools import product

import ray
import train
import aggregator.aggregator as aggregator



def wrapper(config_env, config_agent, config_gridsearch):
    """
    - Create & check files and folders
    - Call the training procedure
    """
    # Decode json files into dict
    dict_env = json.loads(config_env)
    dict_agent = json.loads(config_agent)
    grid_search = json.loads(config_gridsearch)

    # Path to the env and the agent directories
    if not os.path.exists(dict_env["env_dir"]):
        os.makedirs(dict_env["env_dir"])
    env_dir = dict_env["env_dir"] + "/" + dict_env["name"]
    agent_dir = env_dir + "/" + dict_agent["name"]
    # Check if the experience has already been done
    # Make the needed folders
    if os.path.exists(env_dir):
        with open(env_dir + '/' + dict_env["name"] + '.json', 'r') as myfile:
            already_here_env = myfile.read()
        dict_already_here_env = json.loads(already_here_env)
        if dict_env["size"] != dict_already_here_env["size"] \
                or dict_env["manual"] != dict_already_here_env["manual"] \
                or dict_env["random_target"] != dict_already_here_env["random_target"] \
                or dict_env["numObjs"] != dict_already_here_env["numObjs"] \
                or dict_env["COLOR_TO_IDX"] != dict_already_here_env["COLOR_TO_IDX"] \
                or dict_env["TYPE_TO_IDX"] != dict_already_here_env["TYPE_TO_IDX"] \
                or dict_env["oneobject"] != dict_already_here_env["oneobject"] \
                or dict_env["TYPE_TO_IDX"] != dict_already_here_env["TYPE_TO_IDX"]:
            print("Try to override an existing env without the same parameters")
            sys.exit()

        if dict_agent["name"] in os.listdir(env_dir):
            print("Environment & Agent names already used")
            sys.exit()
        else:
            os.mkdir(agent_dir)
    else:
        os.makedirs(agent_dir)
        
    # Duplicate configuration files into model_dir and agent_dir
    with open(env_dir + "/" + dict_env["name"] + ".json", 'w') as outfile:
        json.dump(dict_env, outfile)
    
    with open(agent_dir + '/{}.json'.format(dict_agent["name"]), 'w') as outfile:
        json.dump(dict_agent, outfile)

    with open(agent_dir + '/gridsearch.json', 'w') as outfile:
        json.dump(grid_search, outfile)
    
    dict_agent["agent_dir"] = agent_dir
    # Train the model
    clock = time.time()

    @ray.remote(num_gpus=0.5)
    def training(dict_env, dict_agent):
        return train.training(dict_env=dict_env, dict_agent=dict_agent)

    def cleanstr(string):
        string = string.replace("{", "")
        string = string.replace("}", "")
        string = string.replace(":", "_")
        string = string.replace("'", "")
        string = string.replace(" ", "")
        string = string.replace(",", "_")
        return string

    dicts = []
    # Grid search
    items = sorted(grid_search.items())
    keys, values = zip(*items)
    for ind, seed in enumerate(dict_agent["seed"]):
        for val in product(*values):
            d = dict_agent.copy()
            d.update(dict(zip(keys, val)))
            d["seed"] = seed
            d["agent_dir"] = dict_agent["agent_dir"] + "/run_{}".format(ind + 1) + "/{}".format(cleanstr(str(dict(zip(keys, val)))))
            os.makedirs(d["agent_dir"])
            dicts.append(d)
    ray.get([training.remote(dict_env=dict_env, dict_agent=d) for d in dicts])
    print("Time to train the model {} s".format(round(time.time()-clock, 2)))
    aggregator.wrapper(dict_agent["agent_dir"], output="summary")