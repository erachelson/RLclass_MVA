import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
import solutions.RL1_utility_functions

def print_policy(pi):
    env = gym.make('FrozenLake-v1', render_mode="ansi")
    for row in range(env.unwrapped.nrow):
        for col in range(env.unwrapped.ncol):
            print(actions[pi[to_s(row,col)]], end='')
        print()
    return
