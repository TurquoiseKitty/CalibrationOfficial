# the effect of using MMD loss alone

from src.plot_utils import plot_xy_specifyBound
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func, DEFAULT_layers
import matplotlib.pyplot as plt
from src.models import MC_dropnet
import torch
from src.evaluations import mu_sig_toQuants
from src.losses import MMD_Loss, rmse_loss, gaussian_kernel_func


def CHECK_MMDLoss():

    fig, (ax1, ax2) = plt.subplots(1, 2)


    X_train = np.linspace(0,15,4000)
    Y_train = 3*np.sin(X_train/5) + 4*np.sin(X_train/5)*np.random.randn(4000)

    X_val = np.linspace(0, 15, 1000)
    Y_val = 3*np.sin(X_val/5) + 4*np.sin(X_val/5)*np.random.randn(1000)

    y_pred = 3*np.sin(X_val/5)

    y_UP1 = y_pred + 4*np.sin(X_val/5) * normalZ.ppf(Upper_quant)
    y_LO1 = y_pred + 4*np.sin(X_val/5) * normalZ.ppf(Lower_quant)


    X_train = torch.Tensor(X_train).view(-1, 1).cuda()
    X_val = torch.Tensor(X_val).view(-1, 1).cuda()

    Y_train = torch.Tensor(Y_train).cuda()
    Y_val = torch.Tensor(Y_val).cuda()

    MMD_model = MC_dropnet(
        n_input = 1,
        drop_rate= 0.,
        hidden_layers= [10]
    )

    MMD_model.train(X_train, Y_train, X_val, Y_val,
                bat_size = 100,
                LR = 1E-2,
                N_Epoch = 100,
                early_stopping=False,
                train_loss = MMD_Loss,
                val_loss_criterias = {
                    "MMD" : MMD_Loss,
                    "rmse": rmse_loss
                },
                monitor_name = "MMD"
                )


    output = MMD_model.predict(X_val)

    means = output[:, 0].detach()
    sigs = output[:, 1].detach()

    pred_quants = mu_sig_toQuants(mu = means, sig = sigs, quantiles = [Lower_quant, Upper_quant])

    pred_LO = pred_quants[0].cpu().numpy()
    pred_UP = pred_quants[1].cpu().numpy()

    plot_xy_specifyBound(
        y_pred = y_pred,

        y_UP = y_UP1,
        y_LO = y_LO1,

        y_true = Y_val.cpu().numpy(),
        x = X_val.view(-1).cpu().numpy(),
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax1,
        title = "Confidence Band, Oracle"
    )

    plot_xy_specifyBound(
        y_pred = means.cpu().numpy(),

        y_UP = pred_UP,
        y_LO = pred_LO,

        y_true = Y_val.cpu().numpy(),
        x = X_val.view(-1).cpu().numpy(),
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax2,
        title = "Confidence Band, MMD Loss"

    )

    plt.savefig("Plots_bundle/CHECK/check_MMDLoss.png")

    plt.show(block=True)





if __name__ == "__main__":
    CHECK_MMDLoss()

    '''
    X = np.linspace(0,15,10000)

    tar = 4*np.sin(X/3)*np.random.randn(10000)

    real = 4*np.sin(X/3)*np.random.randn(10000)

    fals = 6*np.sin(X/3)*np.random.randn(10000)

    print("Oracle MMD: ", gaussian_kernel_func(torch.Tensor(tar), torch.Tensor(real)))
    print("Fake MMD: ", gaussian_kernel_func(torch.Tensor(tar), torch.Tensor(fals)))
    '''
