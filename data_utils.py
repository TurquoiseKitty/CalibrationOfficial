from src.DEFAULTS import DEFAULT_mean_func, DEFAULT_hetero_sigma
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import os
import torch


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



def get_uci_data(data_name, dir_name = "Datasets/UCI_datasets"):
    
    data = np.loadtxt("{}/{}.txt".format(dir_name, data_name))
    x_al = data[:, :-1]
    y_al = data[:, -1].reshape(-1)

    return x_al, y_al

def normalize(data):
    
    normalizer = StandardScaler().fit(data)
    
    return normalizer.transform(data), normalizer


def seed_all(seed = 1234):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def splitter(N1, N2, seed = 1234):

    seed_all(seed)

    fullen = N1 + N2

    idx_permu = np.random.choice(fullen, fullen, replace=False)

    return idx_permu[:N1], idx_permu[N1:]


def common_processor_UCI(x, y, recal_percent = 0.1, seed = 1234):

    assert len(x.shape) == 2
    assert len(y.shape) == 1
 
    x_normed, x_normalizer = normalize(x)
    x = x_normed

    N_train = int(len(x) * 0.9)
    N_test = len(x) - N_train

    tr_idx, test_idx = splitter(N_train, N_test, seed = seed)

    # sometimes also leave out some for recalibration
    if recal_percent > 1E-3:

        N_recalibration = int(N_train * recal_percent)
        N_realTrain = N_train - N_recalibration

        tr_idx_idx, recal_idx_idx = splitter(N_realTrain, N_recalibration, seed = seed)

        recal_idx = tr_idx[recal_idx_idx]

        train_idx = tr_idx[tr_idx_idx]

    else:

        # when not enough data, we may also use training set for recalibration

        train_idx = tr_idx

        recal_idx = tr_idx

    
    
    train_X, test_X, recal_X = x[train_idx], x[test_idx], x[recal_idx]

    train_Y, test_Y, recal_Y = y[train_idx], y[test_idx], y[recal_idx]

    return train_X, test_X, recal_X, train_Y, test_Y, recal_Y
