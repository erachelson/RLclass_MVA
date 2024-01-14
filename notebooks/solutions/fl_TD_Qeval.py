import gymnasium as gym
import numpy as np
from tqdm import tqdm

def TD_Qeval(env, pi, max_steps, alpha, gamma, Qinit=None, Qtrue=None):
    error = np.zeros((max_steps))
    if (Qinit is None):
        Qinit = np.zeros((env.observation_space.n, env.action_space.n))
    Q = np.copy(Qinit)
    x,_ = env.reset()
    for t in tqdm(range(max_steps)):
        a = np.random.randint(4)
        y,r,d,_,_ = env.step(a)
        Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][pi[y]]-Q[x][a])
        if(Qtrue is not None):
            error[t] = np.max(np.abs(Q-Qtrue))
        if d==True:
            x,_ = env.reset()
        else:
            x=y
    return Q, error