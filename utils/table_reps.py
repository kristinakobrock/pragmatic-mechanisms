import numpy as np


def calc_mean_std(row):
    mean = np.mean(row)
    std = np.std(row)
    return f'{mean:.2f} ± {std:.2f}'