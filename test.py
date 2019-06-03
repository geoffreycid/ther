import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

# First frames to make a state
# for _ in range(dict_agent["frames"] - 1):
#    action = env.action_space.sample()
#    out_step = env.step(action)
#    observation, reward, terminal = out_step[0], out_step[1], out_step[2]
#    state_frames.append(observation['image'])