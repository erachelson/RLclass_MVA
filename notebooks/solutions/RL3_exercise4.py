### WRITE YOUR CODE HERE
# If you get stuck, uncomment the line above to load a correction in this cell (then you can execute this code).

import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# parameters
gamma = 0.9
lambd = 0.5
alpha = 0.001
max_steps = 2000000
V = np.zeros((env.observation_space.n))
e = np.zeros((env.observation_space.n))

# error plotting
error = np.zeros((max_steps)) # used to track the convergence to Vtrue

x = env.reset()
for t in range(max_steps):
    y,r,d,_ = env.step(fl.RIGHT)
    delta = r+gamma*V[y]-V[x]
    for s in range(env.observation_space.n):
        if s==x:
            e[s] = 1
        else:
            e[s] = e[s]*gamma*lambd
        V[s] = V[s] + alpha * e[s] * delta
    error[t] = np.max(np.abs(V-Vtrue))
    if d==True:
        x = env.reset()
        e = np.zeros((env.observation_space.n))
    else:
        x=y

print(V)
print(Vtrue)
plt.plot(error)
plt.figure()
plt.semilogy(error);