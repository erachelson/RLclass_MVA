import gymnasium as gym
import numpy as np
from tqdm import tqdm

def TD_Veval_stoch_pi(env, pi, max_steps, alpha, gamma, Vinit=None, Vtrue=None, disable_tqdm=False):
    error = np.zeros((max_steps))
    if (Vinit is None):
        Vinit = np.zeros((env.observation_space.n))
    V = np.copy(Vinit)
    x,_ = env.reset()
    for t in tqdm(range(max_steps), disable=disable_tqdm):
        a = np.random.choice(env.action_space.n, p=pi[x,:])
        y,r,d,_,_ = env.step(a)
        aa = np.random.choice(env.action_space.n, p=pi[y,:])
        V[x] = V[x] + alpha * (r+gamma*V[y]-V[x])
        if(Vtrue is not None):
            error[t] = np.max(np.abs(V-Vtrue))
        if d==True:
            x,_ = env.reset()
        else:
            x=y
    return V, error
