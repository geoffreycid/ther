import gym_minigrid.envs.game as game
import numpy as np
import matplotlib.pyplot as plt

dict_env = {
    "name": "5x5-C4-N2-O2",
    "game_type": "fetch",
    "size": 5,
    "numObjs": 2,
    "manual": 0,
    "random_target": 1,
    "oneobject": 0,
    "COLOR_TO_IDX": {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3
    },
    "TYPE_TO_IDX": {
        "key": 0,
        "ball": 1
    },
    "SENIORITY_TO_IDX": {
        'veryyoung': 0,
        'young': 1,
        'normal': 2,
        'old': 3,
        'veryold': 4,
    },
    "T_max": 500,
}

dict_agent = {
    "name": "double-dqn-her",
    "agent": "double-dqn-her",
    "seed": 8,
    "frames": 4,
    "n_keep_correspondence": 1,
    "skew_ratio": 0.5,
    "memory_size": 1e5
}

env = game.game(dict_env, dict_agent)

observation = env.reset()

for _ in range(1000):

    action = env.action_space.sample()
    out = env.step(action)
    #observation, reward, done, info, autre = env.step(action)
    observation = out[0]
    done = out[2]
    im = observation["image"]
    image = env.render("rgb_array")
    #plt.imshow(image)
    #plt.show()
    if im[3, 6, 0] != 1 and not done:
        print("error")
    if done:
        observation = env.reset()

env.close()