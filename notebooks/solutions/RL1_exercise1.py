import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

def mc_eval(pi,nb_trials):
    env = gym.make('FrozenLake-v1', render_mode="ansi")
    horizon = 200
    gamma = 0.9
    Vepisode = np.zeros(nb_trials)
    for i in range(nb_trials):
        env.reset()
        for t in range(horizon):
            next_state, r, done, _, _ = env.step(fl.RIGHT)
            Vepisode[i] += gamma**t * r
            if done:
                break
    return Vepisode
