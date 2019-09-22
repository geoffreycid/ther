import gym_minigrid.envs.game as game
import numpy as np
import matplotlib.pyplot as plt


with open('configs/envs/fetch.json', 'r') as myfile:
    config_env = myfile.read()

#with open('configs/agents/fetch/duelingdoubledqn.json', 'r') as myfile:
#    config_agent = myfile.read()

#with open('configs/experts/expert_to_learn_rnn.json', 'r') as myfile:
#    config_expert = myfile.read()
import json

dict_env = json.loads(config_env)
env = game.game(dict_env)
env.render('rgb_array')
observation = env.reset()

for _ in range(1000):

    action = env.action_space.sample()
    out = env.step(action)
    #observation, reward, done, info, autre = env.step(action)
    observation = out[0]
    done = out[2]
    im = observation["image"]
    #image = env.render("rgb_array")
    #plt.imshow(image)
    #plt.show()
    if done:
        observation = env.reset()

env.close()

dict_words = {
    "start": 0,
    "get": 1,
    "a": 2,
    "go": 3,
    "fetch": 4,
    "you": 5,
    "must": 6,
    "red": 7,
    "green": 8,
    "blue": 9,
    "purple": 10,
    "yellow": 11,
    "grey": 12,
    "key": 13,
    "ball": 14,
    "veryyoung": 15,
    "young": 16,
    "middle": 17,
    "old": 18,
    "veryold": 19,
    "verysmall": 20,
    "small": 21,
    "average": 22,
    "big": 23,
    "verybig": 24,
    ".": 25
}