from solutions.fl_actions import actions
from solutions.fl_to_s import to_s

def print_policy(env,pi):
    for row in range(env.unwrapped.nrow):
        for col in range(env.unwrapped.ncol):
            print(actions[pi[to_s(env,row,col)]], end='')
        print()
    return