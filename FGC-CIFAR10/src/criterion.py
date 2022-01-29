import torch
import torch.nn.functional as F
import torch.nn as nn


## gate contrastive loss
class KNNCriterion(nn.Module):
    def __init__(self, nn_num=5, nn_epoch=20):
        super(KNNCriterion, self).__init__()
        self.nn_num   = nn_num  # k nearest neighbourhood
        self.nn_epoch = nn_epoch # warm

    def forward(self, x, y, epoch):
        eps = 1e-7
        
        batchSize = x.size(0)
        preds = F.softmax(x, 1)

        # no knn
        if epoch <= self.nn_epoch:
            assert y.numel() == batchSize

            pos_idx = y.view(-1, 1)
            x_ans = torch.gather(preds, 1, pos_idx)
            l_x_ans = -1 * torch.log(x_ans.sum(1)+eps).sum(0)
        
        # knn
        else:
            assert y.numel() == batchSize * (self.nn_num + 1)
            #assert y.numel() == batchSize * (self.nn_num)

            nn_idx = y.view(batchSize, -1)
            x_ans_nn = torch.gather(preds, 1, nn_idx)
            l_x_ans =  -1 * torch.log(x_ans_nn.sum(1)+eps).sum(0)

        loss = l_x_ans / batchSize
        return loss