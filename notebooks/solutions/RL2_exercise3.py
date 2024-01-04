import gymnasium as gym
import gymnasium.envs.toy_text.frozen_lake as fl
import numpy as np

env = gym.make('FrozenLake-v1', render_mode="ansi")

gamma = 0.9
r_pi0 = np.zeros((env.observation_space.n))
P_pi0 = np.zeros((env.observation_space.n, env.observation_space.n))
for s in range(env.observation_space.n):
    outcomes = env.unwrapped.P[s][fl.RIGHT]
    for o in outcomes:
        p  = o[0]
        s2 = o[1]
        r  = o[2]
        P_pi0[s][s2] += p
        r_pi0[s] += p*r
I = np.eye(env.observation_space.n)
V_pi0 = np.dot(np.linalg.inv(I - gamma*P_pi0), r_pi0)
print(V_pi0)
