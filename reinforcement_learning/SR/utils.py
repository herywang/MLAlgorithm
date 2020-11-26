import numpy as np

def onehot(value, max_value):
    vec = np.zeros(max_value)
    vec[value] = 1
    return vec

def twohot(value, max_value):
    vec_1 = np.zeros(max_value)
    vec_2 = np.zeros(max_value)
    vec_1[value[0]] = 1
    vec_2[value[1]] = 1
    return np.concatenate([vec_1, vec_2])

def onehot_mat(value, max_value):
    # value: [batch_size, state_value]
    row = value.shape[0]
    vec = np.zeros((row, max_value), dtype=np.float)
    for i in range(row):
        vec[i] = onehot(value[1], max_value)
    return vec

def mask_grid(grid, blocks, mask_value=-100):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if [i,j] in blocks:
                grid[i,j] = mask_value
    grid = np.ma.masked_where(grid == mask_value, grid)
    return grid