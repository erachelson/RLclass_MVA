import numpy as np
from tqdm import tqdm
from solutions.fl_greedyQpolicy import greedyQpolicy

def API_with_TD0(env, nb_iter, td0_steps, init_pi, behavior_pi, alpha, gamma, Qinit=None, Qtrue=None, save_frequency=1, disable_tqdm=False):
    nb_steps = nb_iter*td0_steps
    error = np.zeros((nb_steps))
    save_steps = save_frequency*td0_steps
    if (Qinit is None):
        Qinit = np.zeros((env.observation_space.n, env.action_space.n))
    Q = np.copy(Qinit)
    pi = np.copy(init_pi)
    pi_sequence = [pi]
    x,_ = env.reset()
    for t in tqdm(range(nb_steps), disable=disable_tqdm):
        if ( (t+1) % td0_steps==0):
            pi = greedyQpolicy(env, Q)
        if ( (t+1) % save_steps==0):
            pi_sequence.append(pi)
        a = np.random.choice(env.action_space.n, p=beta[x,:])
        y,r,d,_,_ = env.step(a)
        Q[x][a] = Q[x][a] + alpha * (r+gamma*Q[y][pi[y]]-Q[x][a])
        if(Qtrue is not None):
            error[t] = np.max(np.abs(Q-Qtrue))
        if d==True:
            x,_ = env.reset()
        else:
            x=y
    return pi_sequence,Q,error
