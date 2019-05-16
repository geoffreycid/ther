# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:15:23 2019

@author: geoffreycideron
"""
import os
import sys
import json
import time

import train


def wrapper(config_env, config_agent):
    """
    - Create & check files and folders
    - Call the training procedure
    """
    # Decode json files into dict
    dict_env = json.loads(config_env)
    dict_agent = json.loads(config_agent)
    
    # Path to the env and the agent directories
    if not os.path.exists(dict_env["env_dir"]):
        os.makedirs(dict_env["env_dir"])
    env_dir = dict_env["env_dir"] + "/" + dict_env["name"]
    agent_dir = env_dir + "/" + dict_agent["name"]
    # Check if the experience has already been done
    # Make the needed folders
    if os.path.exists(env_dir):
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
    
    dict_agent["agent_dir"] = agent_dir
    # Train the model
    clock = time.time()
    train.training(dict_env, dict_agent)
    print("Time to train the model {} s".format(round(time.time()-clock, 2)))