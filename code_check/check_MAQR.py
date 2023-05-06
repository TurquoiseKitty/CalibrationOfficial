# we validate the rightness of MAQR algorithm proposed in Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification


from src.plot_utils import plot_xy_specifyBound
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func, DEFAULT_layers
import matplotlib.pyplot as plt
from src.models import vanilla_predNet
import torch
from src.evaluations import mu_sig_toQuants
from src.losses import rmse_loss, mse_loss
from src.kernel_methods import tau_to_quant_datasetCreate
from sklearn.ensemble import RandomForestRegressor



mu_func = lambda x : 3*np.sin(x/5)
sigma_func = lambda x : 4*np.sin(x/5)

def CHECK_MAQR():

    fig, (ax1, ax2) = plt.subplots(1, 2)


    X_train = np.linspace(0,15,4000)
    
    Y_train = mu_func(X_train) + sigma_func(X_train)*np.random.randn(4000)

    X_val = np.linspace(0, 15, 1000)
    Y_val = mu_func(X_val) + sigma_func(X_val)*np.random.randn(1000)

    y_pred = mu_func(X_val)

    y_UP1 = y_pred + sigma_func(X_val) * normalZ.ppf(Upper_quant)
    y_LO1 = y_pred + sigma_func(X_val) * normalZ.ppf(Lower_quant)


    X_train = torch.Tensor(X_train).view(-1, 1).cuda()
    X_val = torch.Tensor(X_val).view(-1, 1).cuda()

    Y_train = torch.Tensor(Y_train).cuda()
    Y_val = torch.Tensor(Y_val).cuda()



    pred_model = vanilla_predNet(
        n_input = 1,
        hidden_layers= [10, 10]
    )

    pred_model.train(X_train, Y_train, X_val, Y_val,
                bat_size = 10,
                LR = 5E-3,
                N_Epoch = 100,
                early_stopping=True,

                train_loss = mse_loss,
                val_loss_criterias = {
                    "mse" : mse_loss,
                    "rmse": rmse_loss
                },
                monitor_name = "mse"

                )
    
    # but after this, we need one more regression

    Z = X_train.cpu()
    eps = (Y_train - pred_model(X_train).view(-1)).detach().cpu()

    reg_X, reg_Y = tau_to_quant_datasetCreate(Z, epsilon=eps, quants= np.linspace(0,1,100),wid = 1E-1)

    depth = 10

    tree_model = RandomForestRegressor(max_depth=depth, random_state=0)
    tree_model.fit(reg_X.detach().cpu().numpy(), reg_Y.detach().cpu().numpy())


    pred_mean = pred_model.predict(X_val).view(-1).cpu().numpy()

    up_quant_Z = np.concatenate([X_val.cpu().numpy(), np.expand_dims(np.array([Upper_quant]*len(X_val)), axis = 1)], axis = 1)

    lo_quant_Z = np.concatenate([X_val.cpu().numpy(), np.expand_dims(np.array([Lower_quant]*len(X_val)), axis = 1)], axis = 1)

    up_epsi = tree_model.predict(up_quant_Z)
    lo_epsi = tree_model.predict(lo_quant_Z)

    

    pred_LO = pred_mean + lo_epsi
    pred_UP = pred_mean + up_epsi

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
        y_pred = pred_mean,

        y_UP = pred_UP,
        y_LO = pred_LO,

        y_true = Y_val.cpu().numpy(),
        x = X_val.view(-1).cpu().numpy(),
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax2,
        title = "Confidence Band, MAQR"

    )

    plt.savefig("Plots_bundle/CHECK/check_MAQR.png")

    plt.show(block=True)





if __name__ == "__main__":
    CHECK_MAQR()

