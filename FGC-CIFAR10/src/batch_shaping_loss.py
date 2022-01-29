import torch
import torch.nn as nn
from torch.autograd import Function
from scipy.stats import beta

class BatchShapingLoss(Function):
    @staticmethod
    def forward(ctx, x):
        x = x.squeeze(-1).squeeze(-1)
        N, M = x.size()

        x_sort, id_sort = torch.sort(x, dim=0)
        _, id_re = torch.sort(id_sort, dim=0)
        
        e_cdf = torch.arange(1, N+1).float()/(N+1)
        e_cdf = e_cdf.unsqueeze(-1).to(x.device)
        
        p_cdf = torch.Tensor(beta.cdf(x_sort.detach().cpu().numpy(), a=0.6, b=0.4)).to(x_sort.device)
        p_pdf = beta.pdf(x_sort.detach().cpu().numpy(), a=0.6, b=0.4)
        ## p_pdf: set 'inf' to 0
        p_pdf[p_pdf == float('inf')] = 0
        p_pdf = torch.Tensor(p_pdf).to(x_sort.device)

        ctx.save_for_backward(id_re, e_cdf, p_cdf, p_pdf)
        return torch.pow(e_cdf - p_cdf, 2).sum() / N

    @staticmethod
    def backward(ctx, grad_output):
        id_re, e_cdf, p_cdf, p_pdf = ctx.saved_tensors
        N = p_cdf.size(0)
        grad = -2 * p_pdf * (e_cdf - p_cdf) / N
        grad_re = torch.zeros_like(grad)
        _, M = grad.size()
        for i in range(M):
            grad_re[:, i] = grad[:, i][id_re[:, i]]
        
        return grad_re.unsqueeze(-1).unsqueeze(-1) * grad_output 
