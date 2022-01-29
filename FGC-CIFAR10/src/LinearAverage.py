import torch
from torch.autograd import Function
from torch import nn
import math

## gate memory bank
class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        # print(x)
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N
        # print(out)

        self.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        # print(x)
        # y = torch.cat((y, y), 0)
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        # print(weight_pos)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        # print(x.data)
        # print('*'*60)
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        
        self.nLem = outputSize
        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        # print(x)
        x = x.squeeze(-1).squeeze(-1)
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


## feature memory bank
class FeatureMemoryBank(nn.Module):
    def __init__(self, inputSize, outputSize, nn_num=5, nn_epoch=20, momentum=0.5):
        super(FeatureMemoryBank, self).__init__()
        self.momentum = momentum
        self.nn_num = nn_num
        self.nn_epoch = nn_epoch # warm
        self.nLem = outputSize
        
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, epoch):
        x = x.squeeze(-1).squeeze(-1).detach()
        nn_epoch = self.nn_epoch
        
        if epoch <= nn_epoch: 
            self._update_bank(x, y)
            return y
        else:
            nn_idx = self._knn(x, y)
            self._update_bank(x, y)    
            return nn_idx
    
    def _update_bank(self, x, y):
        momentum = self.momentum
        
        # update the non-parametric data
        weight_pos = self.memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        self.memory.index_copy_(0, y, updated_weight)
        return None
    
    def _knn(self, x, y):
        nn_num = self.nn_num
        mem = self.memory
        mem_batch = x #mem.index_select(0, y)
        
        sims_batch = torch.mm(mem_batch, mem.t())
        sims_reself = sims_batch.scatter(1, y.view(-1, 1), -2)
        
        reself_nnidx = sims_reself.topk(nn_num, largest=True)[1]
        #return reself_nnidx
        nn_idx = torch.cat((y.view(-1, 1), reself_nnidx), 1)
        return nn_idx

