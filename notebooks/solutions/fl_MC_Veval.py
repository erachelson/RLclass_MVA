import gymnasium as gym
import numpy as np
from tqdm import tqdm

def MC_Veval(env, pi, max_steps, max_episodes, alpha, gamma, Vinit=None, Vtrue=None, disable_tqdm=False):
    error = np.zeros((max_episodes)) # used to track the convergence to V_pi0...
    cumulated_steps = np.zeros((max_episodes)) # ... against the number of samples
    if (Vinit is None):
        Vinit = np.zeros((env.observation_space.n))
    V = np.copy(Vinit)
    for ep in tqdm(range(max_episodes), disable=disable_tqdm):
        x,_ = env.reset()
        episode = []
        # Run episode
        for t in range(max_steps):
            y,r,d,_,_ = env.step(pi[x])
            episode.append([x,r])
            if d==True:
                cumulated_steps[ep] = cumulated_steps[ep-1] + t
                break
            else:
                x=y
        # Update values
        T = len(episode)
        G = np.zeros((T))
        G[-1] = episode[-1][1]
        x = episode[-1][0]
        V[x] = V[x] + alpha * (G[-1] - V[x])
        for t in range(-2,-T-1,-1):
            G[t] = episode[t][1] + gamma*G[t+1]
            x = episode[t][0]
            V[x] = V[x] + alpha * (G[t] - V[x])
        if (Vtrue is not None):
            error[ep] = np.max(np.abs(V-Vtrue))
    return V, cumulated_steps, error
