import torch
import utility as ut

@ut.timer
def sinkhorn_div(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=200, p=2):
    """
    Computes the Sinkhorn divergence between two empirical distributions.

    Args:
        x (torch.Tensor): The first empirical distribution with shape (n, d).
        y (torch.Tensor): The second empirical distribution with shape (m, d).
        alpha (torch.Tensor, optional): Weights for the first distribution. Defaults to uniform weights.
        beta (torch.Tensor, optional): Weights for the second distribution. Defaults to uniform weights.
        epsilon (float, optional): Regularization parameter for the Sinkhorn algorithm. Defaults to 0.01.
        num_iters (int, optional): Maximum number of iterations for the Sinkhorn algorithm. Defaults to 200.
        p (int, optional): The norm degree for computing pairwise distance. Defaults to 2.

    Returns:
        float: The computed Sinkhorn divergence between the two distributions.
    """

    c = torch.cdist(x, y, p=p)
 
    if alpha is None:
        alpha = torch.ones(x.shape[0], dtype=x.dtype, device=x.device) / x.shape[0]

    if beta is None:
        beta = torch.ones(y.shape[0], dtype=y.dtype, device=y.device) / y.shape[0]

    log_alpha = torch.unsqueeze(torch.log(alpha), 1)
    log_beta = torch.log(beta)

    f, g = 0. * alpha, 0. * beta
    f_, iter = 1. * alpha, 0
    while torch.linalg.norm(f - f_, ord=1) / torch.linalg.norm(f_, ord=1) > 1e-3 and iter < num_iters:
        f_ = f
        f = - epsilon * torch.logsumexp(log_beta + (g - c) / epsilon, axis=1)
        g = - epsilon * torch.logsumexp(log_alpha + (torch.unsqueeze(f, 1) - c) / epsilon, axis=0)
        iter += 1
    #print(iter)

    OT_alpha_beta = torch.sum(f * alpha) + torch.sum(g * beta)
    
    c = torch.cdist(x, x, p=p)
    f = 0. * alpha
    f_, iter = 1. * alpha, 0
    log_alpha = torch.squeeze(log_alpha)
    while torch.linalg.norm(f - f_, ord=1) / torch.linalg.norm(f_, ord=1) > 1e-3 and iter < num_iters:
        f_ = f
        f = 0.5 * (f - epsilon * torch.logsumexp(log_alpha + (f - c) / epsilon, axis=1) )
        iter += 1
    #print(iter)

    c = torch.cdist(y, y, p=p)
    g = 0. * beta
    g_, iter = 1. * beta, 0
    while torch.linalg.norm(g - g_, ord=1) / torch.linalg.norm(g_, ord=1) > 1e-3 and iter < num_iters:
        g_ = g
        g = 0.5 * (g - epsilon * torch.logsumexp(log_beta + (g - c) / epsilon, axis=1) )
        iter += 1
    return OT_alpha_beta - torch.sum(f * alpha) - torch.sum(g * beta)
