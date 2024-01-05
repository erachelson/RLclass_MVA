import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np
from solutions.RL2_exercise1 import Q_from_V
from solutions.RL2_exercise2 import greedyQpolicy
from solutions.RL2_utility_functions import print_policy
from solutions.RL1_utility_functions import actions
from solutions.RL2_exercise7 import value_iteration
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode="ansi")
Vinit = np.zeros((env.observation_space.n))
Vstar,residuals = value_iteration(Vinit,1e-4,1000)
Qstar = Q_from_V(Vstar)
print(actions)
print(Qstar)
pi_star = greedyQpolicy(Qstar)
print_policy(pi_star)
