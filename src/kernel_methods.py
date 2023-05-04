import numpy as np
import torch
from torch.linalg import norm
from numpy.linalg import norm as npnorm


def kernel_estimator(
        test_Z: torch.Tensor,
        recal_Z: torch.Tensor,
        recal_epsilon: torch.Tensor,
        quants,
        base_kernel = lambda X : torch.exp(-norm(X, dim = 2) ** 2),
        lamb = 1,
        wid = 1E-1
):
    assert isinstance(test_Z, torch.Tensor)
    assert isinstance(recal_Z, torch.Tensor)
    assert isinstance(recal_epsilon, torch.Tensor)


    assert len(test_Z.shape) == 2
    assert len(recal_Z.shape) == 2
    assert len(recal_epsilon.shape) == 1

    # quants should be a rising list
    assert npnorm(np.sort(quants) - quants) < 1E-6

    quants = torch.Tensor(quants).to(test_Z.device)
    assert len(quants.shape) == 1
    
    sorted_epsi, indices = torch.sort(recal_epsilon, dim = 0)
    sorted_recal_Z = recal_Z[indices]

    test_Z_unsqueezed = test_Z.unsqueeze(1).repeat(1, len(recal_epsilon), 1)
    sorted_recal_Z_unsqueezed = sorted_recal_Z.unsqueeze(0) .repeat(len(test_Z),1,1)

    dist_mat = lamb * base_kernel((sorted_recal_Z_unsqueezed - test_Z_unsqueezed) / wid)

    summation_matform = torch.triu(torch.ones(len(recal_Z), len(recal_Z))).to(test_Z.device)

    aggregated_dist_mat = torch.matmul(dist_mat, summation_matform)

    empirical_quantiles = aggregated_dist_mat / aggregated_dist_mat[:, -1:]


    quantiles_unsquze = empirical_quantiles.view(empirical_quantiles.shape + (-1,))

    tf_mat = quantiles_unsquze <= quants

    harvest_ids = len(recal_Z) - torch.permute(tf_mat.sum(dim=1), (1, 0))

    return sorted_epsi[harvest_ids]

