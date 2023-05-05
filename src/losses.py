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



def mean_std_forEnsemble(predic, target):

    assert len(predic) % len(target) == 0

    n_mod = int(len(predic) / len(target))

    target = target.repeat(n_mod)

    return mean_std_norm_loss(predic, target)



def gaussian_kernel_func(
    source,
    target,
    bandwidth_list = [1, 4, 8, 16, 32]
):
    assert isinstance(source, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert len(source.shape) == 1 

    n_samples = len(source) + len(target)

    source = source.view(-1,1)
    target = target.view(-1,1)

    total = torch.cat([source, target], dim = 0)


    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_dist = ((total0 - total1)**2).sum(dim = 2)

    kernel_val =  [torch.exp(-L2_dist / wid) for wid in bandwidth_list]

    kernel_val = sum(kernel_val)

    XX = kernel_val[:len(source), :len(source)]

    XY = kernel_val[:len(source), len(source):]

    YX = kernel_val[len(source):, :len(source)]

    YY = kernel_val[len(source):, len(source):]
    	    
    loss = torch.mean(XX + YY - XY -YX)

    return loss


def MMD_Loss(source, target, bandwidth_list = [1, 4, 8, 16, 32]):


    source_len = len(source)

    with torch.no_grad():

        errs = torch.randn_like(target).to(source.device)

    real_source = source[:, 0] + errs * source[:, 1]

    loss = gaussian_kernel_func(
        real_source,
        target,
        bandwidth_list
    )

    
    return loss

        

 



