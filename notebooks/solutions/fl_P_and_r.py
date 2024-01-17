import numpy as np

def fl_P_and_r(env,pi):
    r_pi = np.zeros((env.observation_space.n))
    P_pi = np.zeros((env.observation_space.n, env.observation_space.n))
    for x in range(env.observation_space.n):
        outcomes = env.unwrapped.P[x][pi[x]]
        for o in outcomes:
            p = o[0]
            y = o[1]
            r = o[2]
            P_pi[x,y] += p
            r_pi[x] += r*p
    return P_pi,r_pi
