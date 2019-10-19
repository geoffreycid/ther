from train import training

path = "/home/gcideron/ther/"

with open(path + 'configs/envs/fetch.json', 'r') as myfile:
    config_env = myfile.read()

with open(path + 'configs/agents/fetch/duelingdoubledqn.json', 'r') as myfile:
    config_agent = myfile.read()

with open(path + 'configs/experts/expert_to_learn_rnn.json', 'r') as myfile:
    config_expert = myfile.read()
import json

dict_env = json.loads(config_env)
dict_agent = json.loads(config_agent)
dict_agent["agent_dir"] = dict_env["env_dir"] + "/" + dict_env["name"] + "/" + dict_agent["name"]
dict_expert = json.loads(config_expert)
print("Training in progress")
training(dict_env, dict_agent, dict_expert)
