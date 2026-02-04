import numpy as np

def display_function_of_state(f):
    """Plots values of frozen lake states in a 4x4 matrix instead of a vector"""
    print(np.reshape(f, (4,4)))
    return
