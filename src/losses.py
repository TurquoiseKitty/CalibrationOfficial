import torch.nn as nn
import torch

def mse_loss(predic, target):

    if len(predic.shape) != 1:
        predic = predic[:, 0]
       
    assert len(target.shape) == 1

    loss_layer = nn.MSELoss()

    return loss_layer(predic, target) 

def rmse_loss(predic, target):

    return torch.sqrt(mse_loss(predic, target))

def mean_std_norm_loss(predic, target):

    assert len(target.shape) == 1
    assert len(predic.shape) == 2

    mu = predic[:, 0]
    sigma = predic[:,1]

    loss = nn.GaussianNLLLoss()
    return loss(mu, target, sigma**2)