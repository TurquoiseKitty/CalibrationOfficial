from src.plot_utils import plot_xy_specifyBound
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func, DEFAULT_layers
import matplotlib.pyplot as plt
from src.models import MC_dropnet, quantile_predNet
import torch
from src.evaluations import mu_sig_toQuants
from src.losses import MMD_Loss, rmse_loss, avg_pinball_quantile

mu_func = lambda x : 3*np.sin(x/5)
sigma_func = lambda x : 4*np.sin(x/5)

def CHECK_avgPinBall():

    fig, (ax1, ax2) = plt.subplots(1, 2)


    X_train = np.linspace(0,15,10000)
    Y_train = mu_func(X_train) + sigma_func(X_train)*np.random.randn(10000)

    X_val = np.linspace(0, 15, 1000)
    Y_val = mu_func(X_val) + sigma_func(X_val)*np.random.randn(1000)

    y_pred = mu_func(X_val)

    y_UP1 = y_pred + sigma_func(X_val) * normalZ.ppf(Upper_quant)
    y_LO1 = y_pred + sigma_func(X_val) * normalZ.ppf(Lower_quant)


    X_train = torch.Tensor(X_train).view(-1, 1).cuda()
    X_val = torch.Tensor(X_val).view(-1, 1).cuda()

    Y_train = torch.Tensor(Y_train).cuda()
    Y_val = torch.Tensor(Y_val).cuda()

    PinBall_model = quantile_predNet(
        n_input = 1,
        
        hidden_layers= [10, 10],
        n_output= 3
    )

    PinBall_model.train(X_train, Y_train, X_val, Y_val,
                bat_size = 20,
                LR = 5E-3,
                N_Epoch = 100,
                early_stopping=False,
                train_loss = lambda x, y: avg_pinball_quantile(x, y, q_list= [Lower_quant, 0.5, Upper_quant]),
                val_loss_criterias = {
                    "pinball" : lambda x, y: avg_pinball_quantile(x, y, q_list= [Lower_quant, 0.5, Upper_quant]),
                    "rmse": lambda x, y: rmse_loss(x[:,1], y)
                },
                monitor_name = "pinball",

                )


    output = PinBall_model.predict(X_val)

    means = output[:, 1].detach()
    

    pred_LO = output[:, 0].cpu().numpy()
    pred_UP = output[:, 2].cpu().numpy()

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
        title = "Confidence Band, PinBall Loss"

    )

    plt.savefig("Plots_bundle/CHECK/check_avgPinBall_quant.png")

    plt.show(block=True)





if __name__ == "__main__":
    CHECK_avgPinBall()
