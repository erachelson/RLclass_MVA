import numpy as np
from tqdm import tqdm
from solutions.fl_epsilon_greedy import epsilon_greedy

def Qlearning(env, max_steps, alpha, gamma, epsilon_update_period, Qinit=None, Qtrue=None, disable_tqdm=False):
    if (Qinit is None):
        Qinit = np.zeros((env.observation_space.n,env.action_space.n)) 
    Qql = np.copy(Qinit)
    epsilon = 1
    error = np.zeros((max_steps))
    x,_ = env.reset()
    for t in tqdm(range(max_steps), disable=disable_tqdm):
        if((t+1)%epsilon_update_period==0):
            epsilon = epsilon/2
        a = epsilon_greedy(env,Qql,x,epsilon)
        y,r,d,_,_ = env.step(a)
        Qql[x][a] = Qql[x][a] + alpha * (r+gamma*np.max(Qql[y][:])-Qql[x][a])
        error[t] = np.max(np.abs(Qql-Qstar))
        if d==True:
            x,_ = env.reset()
        else:
            x=y
    return Qql, error
