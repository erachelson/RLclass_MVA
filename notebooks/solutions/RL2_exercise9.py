import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from solutions.RL2_exercise1 import Q_from_V
from solutions.RL2_exercise2 import greedyQpolicy
from solutions.RL2_exercise5 import policy_eval_iter_mat

def modified_policy_iteration(pi0,m,max_iter):
    env = gym.make('FrozenLake-v1', render_mode="ansi")
    policies = np.zeros((max_iter, env.observation_space.n))
    policies[0,:] = np.copy(pi0)
    for i in range(max_iter-1):
        Vpi = policy_eval_iter_mat(policies[i],m)
        Qpi = Q_from_V(Vpi)
        policies[i+1,:] = greedyQpolicy(Qpi)
        if np.array_equal(policies[i,:],policies[i+1,:]):
            policies = policies[:i,:]
            break
    return policies
