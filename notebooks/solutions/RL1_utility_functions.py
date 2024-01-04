import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl

actions = {fl.LEFT: '\u2190', fl.DOWN: '\u2193', fl.RIGHT: '\u2192', fl.UP: '\u2191'}

def to_s(row,col):
    env = gym.make('FrozenLake-v1', render_mode="ansi")
    return row*env.unwrapped.ncol+col

def to_row_col(s):
    env = gym.make('FrozenLake-v1', render_mode="ansi")
    col = s%env.unwrapped.ncol
    row = int((s-col)/env.unwrapped.ncol)
    return row,col
