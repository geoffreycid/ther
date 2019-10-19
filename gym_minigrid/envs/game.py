#from gym_minigrid.minigrid import *
from gym_minigrid.register import register

#import gym_minigrid.envs.fetchworegister as fetch
import gym_minigrid.envs.fetchattrib as fetch
import gym_minigrid.envs.emptyworegister as empty
import gym


def game(dict_env):
    """
    Create either empty or fetch env
    For fetch: dict_env has to have "game_type", "size", and "numObjs"
    For empty: dict_env has to have "game_type", and "size"
    :param dict_env:
    :return:
    """
    if dict_env["game_type"] == "fetch":
        class FetchGame(fetch.FetchGame):
            def __init__(self):
                super().__init__(color_to_idx=dict_env["COLOR_TO_IDX"],
                                 shade_to_idx=dict_env["SHADE_TO_IDX"],
                                 size_to_idx=dict_env["SIZE_TO_IDX"],
                                 size=dict_env["size"], numObjs=dict_env["numObjs"],
                                 oneobject=dict_env["oneobject"],
                                 random_target=dict_env["random_target"],
                                 reward_if_wrong_object=dict_env["reward_if_wrong_object"],
                                 wrong_object_terminal=dict_env["wrong_object_terminal"],
                                 use_defined_missions=dict_env["use_defined_missions"],
                                 shuffle_attrib=dict_env["shuffle_attrib"],
                                 missions=dict_env["missions"])
        name = 'MiniGrid-Fetch-{}x{}-N{}-v0'.format(dict_env["size"], dict_env["size"], dict_env["numObjs"])
        register(
            id=name,
            entry_point='gym_minigrid.envs:FetchGame'
        )
        return gym.make(name, color_to_idx=dict_env["COLOR_TO_IDX"],
                        shade_to_idx=dict_env["SHADE_TO_IDX"],
                        size_to_idx=dict_env["SIZE_TO_IDX"],
                        size=dict_env["size"], numObjs=dict_env["numObjs"],
                        manual=dict_env["manual"], oneobject=dict_env["oneobject"],
                        random_target=dict_env["random_target"],
                        reward_if_wrong_object=dict_env["reward_if_wrong_object"],
                        wrong_object_terminal=dict_env["wrong_object_terminal"],
                        use_defined_missions=dict_env["use_defined_missions"],
                        shuffle_attrib=dict_env["shuffle_attrib"],
                        missions=dict_env["missions"])

    if dict_env["game_type"] == "empty":
        class EmptyGame(empty.EmptyGame):
            def __init__(self):
                super().__init__(size=dict_env["size"])
        if dict_env["random"]:
            name = 'MiniGrid-Empty-Random-{}x{}-v0'.format(dict_env["size"], dict_env["size"])
            register(
                id=name,
                entry_point='gym_minigrid.envs:EmptyGame'
            )
            return gym.make(name, size=dict_env["size"], agent_start_pos=None)
        else:
            name = 'MiniGrid-Empty-{}x{}-v0'.format(dict_env["size"], dict_env["size"])
            register(
                id=name,
                entry_point='gym_minigrid.envs:EmptyGame'
            )
            return gym.make(name, size=dict_env["size"])


#dict_env = {
#    "game_type": "fetch",
#    "size": 5,
#    "random": 1,
#    "numObjs": 2,
#    "OneObj": 0,
#    "COLOR_TO_IDX": {
#    "red": 0,
#},
#    "manual": 1
#}


import time
#env = game(dict_env)
#obs = env.reset()
#for _ in range(100):
#    curr_obs = obs.copy()
#    obs = env.reset()
#    env.render()
#    print(curr_obs["image"] == obs["image"])
    #env.step(env.action_space.sample()) # take a random action
    #time.sleep()
#env.close()

#print(env)
