import numpy as np

def policy_eval_iter_mat2(env, pi, epsilon, max_iter):
    gamma = 0.9
    # build r and P
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
    # Compute V
    V = np.zeros((env.observation_space.n))
    W = np.zeros((env.observation_space.n))
    residuals = np.zeros((max_iter))
    for i in range(max_iter):
        W = r_pi + gamma * np.dot(P_pi, V)
        residuals[i] = np.max(np.abs(W-V))
        V[:] = W
        if residuals[i]<epsilon:
            residuals = residuals[:i+1]
            break
    return V, residuals