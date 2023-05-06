# This experiment is a mistake, as avgpinball loss can not be used for training the heteroskedastic model

from src.plot_utils import plot_xy_specifyBound
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func, DEFAULT_layers
import matplotlib.pyplot as plt
from src.models import MC_dropnet, quantile_predNet
import torch
from src.evaluations import mu_sig_toQuants
from src.losses import MMD_Loss, rmse_loss, BeyondPinball_quantile, avg_pinball_quantile


def unstable_update():

    sigma = 10
    MID = 0.8

    X_train = np.random.binomial(1, 0.7, 100000)
    Y_train = (1+10 * X_train)* np.random.randn(100000)

    X_val = np.random.binomial(1, 0.7, 1000)
    Y_val = (1+10 * X_val)* np.random.randn(1000)


    X_train = torch.Tensor(X_train).view(-1, 1).cuda()
    X_val = torch.Tensor(X_val).view(-1, 1).cuda()

    Y_train = torch.Tensor(Y_train).cuda()
    Y_val = torch.Tensor(Y_val).cuda()


    BeyondPinBall_model = quantile_predNet(
        n_input = 1,
        
        hidden_layers= [],
        n_output= 1
    )

    BeyondPinBall_model.train(X_train, Y_train, X_val, Y_val,
                bat_size = 200,
                LR = 5E-2,
                N_Epoch = 20,
                early_stopping=False,
                train_loss = lambda x, y: BeyondPinball_quantile(x, y, q_list= [MID]),
                
                val_loss_criterias = {
                    "beyond_pinball" : lambda x, y: BeyondPinball_quantile(x, y, q_list= [MID]),
                    "rmse": lambda x, y: rmse_loss(x, y)
                },
                monitor_name = "beyond_pinball",

                )
    
    avg_PinBall_model = quantile_predNet(
        n_input = 1,
        
        hidden_layers= [],
        n_output= 1
    )

    avg_PinBall_model.train(X_train, Y_train, X_val, Y_val,
                bat_size = 200,
                LR = 5E-2,
                N_Epoch = 20,
                early_stopping=False,
                train_loss = lambda x, y: avg_pinball_quantile(x, y, q_list= [MID]),
                
                val_loss_criterias = {
                    "beyond_pinball" : lambda x, y: avg_pinball_quantile(x, y, q_list= [MID]),
                    "rmse": lambda x, y: rmse_loss(x, y)
                },
                monitor_name = "beyond_pinball",

                )


    to_test = torch.Tensor([
        0, 1
    ]).view(-1, 1).cuda()
    print("Beyond pinball: ",BeyondPinBall_model.predict(to_test).view(-1).detach().cpu().numpy())
    print("Average pinball: ",avg_PinBall_model.predict(to_test).view(-1).detach().cpu().numpy())
    print("real: ", (normalZ.ppf(MID), 11 * normalZ.ppf(MID)))

    





if __name__ == "__main__":
    unstable_update()
