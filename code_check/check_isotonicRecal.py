from src.plot_utils import plot_calibration
from src.isotonic_recal import iso_recal
from src.evaluations import obs_vs_exp, mu_sig_toQuants
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func
import matplotlib.pyplot as plt
import torch


def CHECK_isotonicRecal():

    x = np.linspace(0,15,1000)

    y_pred = DEFAULT_mean_func(x)


    y2 = DEFAULT_mean_func(x) + 2 * np.random.randn(1000)   
    
    exp = np.linspace(0, 1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # we are first making a false estimation. The std of noise is 2, but we mistake it for 1

    # pred_mat = torch.Tensor(normalZ.ppf(exp)).unsqueeze(dim = 1).repeat(1, len(y2))
    pred_mat = mu_sig_toQuants(torch.zeros(len(y2)),torch.ones(len(y2)),exp)

    exp_quants, obs_quants = obs_vs_exp(y_true = torch.Tensor(y2-y_pred), exp_quants = exp, quantile_preds = pred_mat)


    plot_calibration(
        exp_props = exp_quants.cpu().numpy(),
        obs_props = obs_quants.cpu().numpy(),
        ax = ax1,
        title = "False Prediction"
    )

    # then we run isotonic regression

    recalibrator = iso_recal(exp_quants.cpu().numpy(), obs_quants.cpu().numpy())

    reshaped_exp = recalibrator.predict(exp)

    # pred_mat2 = torch.Tensor(normalZ.ppf(reshaped_exp)).unsqueeze(dim = 1).repeat(1, len(y2))

    pred_mat2 = mu_sig_toQuants(torch.zeros(len(y2)),torch.ones(len(y2)),reshaped_exp)

    _, new_obs_quants = obs_vs_exp(y_true = torch.Tensor(y2-y_pred), exp_quants = reshaped_exp, quantile_preds = pred_mat2)


    plot_calibration(
        exp_props = exp_quants.cpu().numpy(),
        obs_props = new_obs_quants.cpu().numpy(),
        ax = ax2,
        title = "Prediction after Isotonic Regression"
    )

    fig.tight_layout(pad=2.0)

    plt.show(block=True)




if __name__ == "__main__":

    CHECK_isotonicRecal()

