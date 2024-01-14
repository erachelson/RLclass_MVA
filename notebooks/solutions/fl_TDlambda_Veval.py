import gymnasium as gym
import numpy as np
from tqdm import tqdm

def TDlambda_Veval(env, pi, max_steps, alpha, gamma, lambd, Vinit=None, Vtrue=None):
    error = np.zeros((max_steps))
    if (Vinit is None):
        Vinit = np.zeros((env.observation_space.n))
    V = np.copy(Vinit)
    e = np.zeros((env.observation_space.n))
    x,_ = env.reset()
    for t in tqdm(range(max_steps)):
        y,r,d,_,_ = env.step(pi[x])
        delta = r+gamma*V[y]-V[x]
        for s in range(env.observation_space.n):
            if s==x:
                e[s] = 1
            else:
                e[s] = e[s]*gamma*lambd
            V[s] = V[s] + alpha * e[s] * delta
        if(Vtrue is not None):
            error[t] = np.max(np.abs(V-Vtrue))
        if d==True:
            x,_ = env.reset()
            e = np.zeros((env.observation_space.n))
        else:
            x=y
    return V, error
