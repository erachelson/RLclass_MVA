import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

def greedyQpolicy(Q):
    env = gym.make('FrozenLake-v1', render_mode="ansi")
    pi = np.zeros((env.observation_space.n),dtype=int)
    for s in range(env.observation_space.n):
        pi[s] = np.argmax(Q[s,:])
    return pi
