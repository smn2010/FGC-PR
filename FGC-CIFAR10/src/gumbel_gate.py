import torch
import torch.nn as nn
from torch.autograd import Function
from collections import OrderedDict
from .batch_shaping_loss import BatchShapingLoss
import torch.nn.functional as F

from src.LinearAverage import LinearAverage, FeatureMemoryBank
from src.criterion import KNNCriterion


class Gumbel(Function):
    @staticmethod
    def forward(ctx, log_alpha, temperature):
        sz = log_alpha.size()
        #U = torch.distributions.Uniform(torch.zeros(sz), torch.ones(sz))
        u = torch.zeros(sz).uniform_()#U.sample()
        l = torch.log(u)-torch.log(1-u)
        l = l.to(log_alpha.device)
        temperature = temperature.to(log_alpha.device)
        y = (log_alpha + l) / temperature

        f = torch.sigmoid(y)
        ctx.save_for_backward(f, temperature)
        ## the unit step function(argmax for binary) in forward.
        m = (y > 0).float()
        return m
    
    @staticmethod
    def backward(ctx, grad_output):
        f, temperature = ctx.saved_tensors
        ## sigmoid in backward.
        grad =  f * (1-f) / temperature
        grad_input = grad_output * grad
        return grad_input, None

class Gate(nn.Module):
    '''Gating module for channel gated networks.
    '''
    def __init__(self, in_channels, out_channels, hidden_channels=16, temperature=2./3):
        super(Gate, self).__init__()
        i_c  = in_channels
        o_c  = out_channels
        h_c  = hidden_channels

        self.t = torch.Tensor([temperature])

        self.mlp = nn.Sequential(OrderedDict([
          ('avgpool', nn.AdaptiveAvgPool2d(1)),
          ('fc1',  nn.Conv2d(i_c, h_c, 1, bias=False)),
          ('bn',   nn.BatchNorm2d(h_c)),
          ('relu', nn.ReLU()),
          ('fc2',  nn.Conv2d(h_c, o_c, 1))
        ]))
        ## The length of the gate vector.
        self.n_dim = o_c
    
    def forward(self, x):
        self.log_alpha = self.mlp(x)
        self.gate_probs = torch.sigmoid(self.log_alpha)
        #self.gate = Gumbel.apply(self.log_alpha, self.t)
        if self.training:
            self.gate = Gumbel.apply(self.log_alpha, self.t)
        else:
            self.gate = (self.gate_probs > 0.5).float()
        return self.gate
    
    
    #############---contrastive loss functions---#############
    def _loss_ct(self, feats, indexes, epoch):
        '''Set positive & negative keys by indexing queue with pseudo_labels.
            Then compute the contrastive loss.
        '''
        nn_idx  = self.feat_mb(feats, indexes, epoch)
        outputs = self.gate_mb(self.log_alpha, indexes)
        loss    = self.criterion(outputs, nn_idx, epoch)
        return loss

    