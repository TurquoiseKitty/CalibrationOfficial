from src.plot_utils import plot_calibration
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func
import matplotlib.pyplot as plt
from src.evaluations import obs_vs_exp
import torch


def CHECK_obs_exp():

    x = np.linspace(0,15,1000)

    y_pred = DEFAULT_mean_func(x)

    y = DEFAULT_mean_func(x) + 1 * np.random.randn(1000)


    exp = np.linspace(0, 1, 10)


    exp_quants, obs_quants = obs_vs_exp(y_true = torch.Tensor(y-y_pred), exp_quants = exp, quantile_preds = torch.Tensor(normalZ.ppf(exp)).unsqueeze(dim = 1).repeat(1, len(y)))

    fig, ax = plt.subplots()

    plot_calibration(
        exp_props = exp_quants.cpu().numpy(),
        obs_props = obs_quants.cpu().numpy(),
        ax = ax
    )

    plt.savefig("Plots_bundle/CHECK/check_evaluationOBEXP.png")

    plt.show(block=True)



if __name__ == "__main__":

    CHECK_obs_exp()

