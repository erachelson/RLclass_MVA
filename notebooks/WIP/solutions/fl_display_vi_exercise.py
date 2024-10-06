import gymnasium as gym
import numpy as np
from solutions.fl_Q_from_V import Q_from_V
from solutions.fl_greedyQpolicy import greedyQpolicy
from solutions.fl_print_policy import print_policy
from solutions.fl_actions import actions
from solutions.fl_value_iteration import value_iteration
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode="ansi")
Vinit = np.zeros((env.observation_space.n))
Vstar,residuals = value_iteration(env,Vinit,1e-4,1000)
Qstar = Q_from_V(env,Vstar)
print(actions)
print(Qstar)
pi_star = greedyQpolicy(env,Qstar)
print_policy(env,pi_star)