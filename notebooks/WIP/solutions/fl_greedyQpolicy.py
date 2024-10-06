import numpy as np

def greedyQpolicy(env,Q):
    pi = np.zeros((env.observation_space.n),dtype=int)
    for s in range(env.observation_space.n):
        pi[s] = np.argmax(Q[s,:])
    return pi