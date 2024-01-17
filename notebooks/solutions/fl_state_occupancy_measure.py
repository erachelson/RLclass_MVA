from solutions.fl_P_and_r import fl_P_and_r

def state_occupancy_measure(P_pi,pi,rho,gamma,horizon):
    state_proba_at_t = rho
    rho_pi = rho
    for i in range(horizon):
        state_proba_at_t = state_proba_at_t @ P_pi
        rho_pi += gamma**(i+1) * state_proba_at_t
    return rho_pi
