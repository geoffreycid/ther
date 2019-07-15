import gym
import gym_minigrid

import numpy as np
import matplotlib.pyplot as plt

import time

env = gym.make("MiniGrid-Fetch-5x5-N2-v0")

observation = env.reset()

for _ in range(10000):

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    im = observation["image"]
    image = env.render("rgb_array")
    #plt.imshow(image)
    #plt.show()
    #time.sleep(0.1)
    if im[3, 6, 0] != 1 and not done:
        print("error")
    if done:
        observation = env.reset()

env.close()