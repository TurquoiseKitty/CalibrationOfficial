from src.DEFAULTS import DEFAULT_mean_func, DEFAULT_hetero_sigma
import numpy as np



def GENERATE_hetero_noise(
    n_points: int = 1000,
    mean_func = DEFAULT_mean_func,
    std_fun = DEFAULT_hetero_sigma
):
    
    bounds = [0, 15]

    x = np.linspace(bounds[0], bounds[1], n_points)

    f = mean_func(x)
    std = std_fun(x)
    noise = np.random.normal(scale=std)
    y = f + noise
    return f, std, y, x