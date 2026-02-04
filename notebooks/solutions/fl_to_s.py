def to_s(env,row,col) -> int:
    """Convert coordinates to state index"""
    return row*env.unwrapped.ncol+col
