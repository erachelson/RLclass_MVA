from solutions.fl_P_and_r import fl_P_and_r

def state_occupancy_measure(P_pi,rho,gamma,horizon):
    state_proba_at_t = rho
    rho_pi = rho
    for t in range(1,horizon+1):
        state_proba_at_t = state_proba_at_t @ P_pi
        rho_pi += gamma**t * state_proba_at_t
    return rho_pi
