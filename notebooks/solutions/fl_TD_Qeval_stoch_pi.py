import gymnasium as gym
import numpy as np
from tqdm import tqdm

def TD_Qeval_stoch_pi(env, pi, beta=None, max_steps=int(1e6), alpha=0.001, gamma=0.9, Qinit=None, Qtrue=None, disable_tqdm=False):
    error = np.zeros((max_steps))
    if (beta is None):
        beta = (1./env.action_space.n) * np.ones((env.observation_space.n,env.action_space.n))
    if (Qinit is None):
        Qinit = np.zeros((env.observation_space.n, env.action_space.n))
    Q = np.copy(Qinit)
    x,_ = env.reset()
    for t in tqdm(range(max_steps), disable=disable_tqdm):
        a = np.random.choice(env.action_space.n, p=beta[x,:])
        y,r,d,_,_ = env.step(a)
        aa = np.random.choice(env.action_space.n, p=pi[y,:])
        Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][aa]-Q[x][a])
        if(Qtrue is not None):
            error[t] = np.max(np.abs(Q-Qtrue))
        if d==True:
            x,_ = env.reset()
        else:
            x=y
    return Q, error
