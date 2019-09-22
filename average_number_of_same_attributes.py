import gym_minigrid.envs.game as game
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_palette("colorblind")


with open('configs/envs/fetch.json', 'r') as myfile:
    config_env = myfile.read()

import json

dict_env = json.loads(config_env)
env = game.game(dict_env)
env.render('rgb_array')
observation = env.reset()

same_attrib_accuracy = []
for _ in range(10):
    env.reset()
    objs = env.objs
    for ind, curr_obj in enumerate(objs):
        objs_without_current_obj = [objct for i, objct in enumerate(objs) if i != ind]
        for obj in objs_without_current_obj:
            accuracy = (curr_obj.color == obj.color) + (curr_obj.size == obj.size) \
                       + (curr_obj.shade == obj.shade) + (curr_obj.type == obj.type)
            accuracy /= 4
            same_attrib_accuracy.append(accuracy)

hist = np.histogram(same_attrib_accuracy, bins=[0, 0.24, 0.26, 0.51, 0.76, 1.1])[0]
hist = hist / hist.sum()
plt.bar(["0", "1", "2", "3", "4"], hist)
plt.title("Average shared attributes between two objects")
plt.xlabel("attributes")
plt.ylabel("")
plt.show()


discriminant_attrib_accuracy = []
for _ in range(1000):
    env.reset()
    objs = env.objs
    curr_obj = objs[0]
    discri = 4
    for obj in objs[1:]:
        shared_attrib = (curr_obj.color == obj.color) + (curr_obj.size == obj.size) \
                   + (curr_obj.shade == obj.shade) + (curr_obj.type == obj.type)
        discri = min(discri, 4-shared_attrib)
    discriminant_attrib_accuracy.append(discri)


hist = np.histogram(discriminant_attrib_accuracy, bins=[0, 1, 2, 3, 4, 5])[0]
hist = hist / hist.sum()
plt.bar(["0", "1", "2", "3", "4"], hist)
plt.title("Discriminant")
plt.xlabel("attributes")
plt.ylabel("")
plt.show()
