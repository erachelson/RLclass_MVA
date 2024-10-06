import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from solutions.fl_Q_from_V import Q_from_V
from solutions.fl_greedyQpolicy import greedyQpolicy
from solutions.fl_policy_eval_iter import policy_eval_iter_mat

def modified_policy_iteration(env,pi0,m,max_iter):
    policies = np.zeros((max_iter, env.observation_space.n))
    policies[0,:] = np.copy(pi0)
    for i in range(max_iter-1):
        Vpi = policy_eval_iter_mat(env,policies[i],m)
        Qpi = Q_from_V(env,Vpi)
        policies[i+1,:] = greedyQpolicy(env,Qpi)
        if np.array_equal(policies[i,:],policies[i+1,:]):
            policies = policies[:i,:]
            break
    return policies