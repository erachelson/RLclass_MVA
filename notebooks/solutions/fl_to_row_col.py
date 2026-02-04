def to_row_col(env,s):
    """Convert state index to coordinates"""
    col = s%env.unwrapped.ncol
    row = int((s-col)/env.unwrapped.ncol)
    return row,col
