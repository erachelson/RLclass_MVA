import numpy as np
from tqdm import tqdm
from solutions.fl_epsilon_greedy import epsilon_greedy

def SARSA(env, max_steps, alpha, gamma, epsilon_update_period, Qinit=None, Qtrue=None, disable_tqdm=False):
    if (Qinit is None):
        Qinit = np.zeros((env.observation_space.n,env.action_space.n)) 
    Q = np.copy(Qinit)
    epsilon = 1
    error = np.zeros((max_steps))
    x,_ = env.reset()
    a = epsilon_greedy(env,Q,x,epsilon)
    for t in tqdm(range(max_steps), disable=disable_tqdm):
        if((t+1)%epsilon_update_period==0):
            epsilon = epsilon/2
        y,r,d,_,_ = env.step(a)
        aa = epsilon_greedy(env,Q,y,epsilon)
        Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][aa]-Q[x][a])
        error[t] = np.max(np.abs(Q-Qstar))
        if d==True:
            x,_ = env.reset()
            a=epsilon_greedy(env,Q,x,epsilon)
        else:
            x=y
            a=aa
    return Q, error
