import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

def mc_eval(env,pi,nb_trials):
    horizon = 200
    gamma = 0.9
    Vepisode = np.zeros(nb_trials)
    for i in range(nb_trials):
        state,_ = env.reset()
        for t in range(horizon):
            next_state, r, done, _, _ = env.step(pi[state])
            Vepisode[i] += gamma**t * r
            state = next_state
            if done:
                break
    return Vepisode
