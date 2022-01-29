import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import time
import math

from src.batch_shaping_loss import BatchShapingLoss
from src.gumbel_gate import Gate
from src.LinearAverage import LinearAverage, FeatureMemoryBank
from src.criterion import KNNCriterion


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

#############--- Basic buiding block ---#############
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.in_planes = in_planes
        
        self.gate = Gate(in_planes, planes)
        ## whether contrast or not.
        self.use_contrast = False
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        ## current features.
        if self.use_contrast: 
            self.avgfeat = self.avgpool(out)
        
        g = self.gate(x)
        out = out * g

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

#############--- Residual network ---#############
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        ## list of gating modules.
        self.gating_modules = []
        for m in self.modules():
            if isinstance(m, Gate): self.gating_modules.append(m)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    
    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        ##
        #self.last_avg_feat = out.detach().cpu()
        ##
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    #############---baseline functions---#############
    def get_params(self):
        '''Return parameters of gating modules and backbone respectively.
        '''
        gate_params = []
        ignored_params = []
        for m in self.modules():
            if isinstance(m, Gate):
                ignored_params += list(map(id, m.parameters()))
                gate_params += m.parameters()
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return gate_params, base_params
    
    def get_loss_bs(self):
        '''Summation of the batch shaping loss of gating modules in the network.
        '''
        loss = 0.
        for g_m in self.gating_modules:
            loss += BatchShapingLoss.apply(g_m.gate_probs)
        return loss

    def get_loss_l0(self):
        '''Summation of the l0 loss of gating modules in the network.
        '''
        loss = 0.
        n_gate = 0
        for g_m in self.gating_modules:
            loss   += g_m.gate_probs.sum()
            n_gate += g_m.gate_probs.numel()
        return loss / n_gate 

    def sparsity(self):
        ''' The on-the-fly gate sparsity of network.
        '''
        g_sum = 0.
        g_num = 0.
        for g_m in self.gating_modules:
            gates =  g_m.gate.detach().cpu()
            g_sum += gates.sum()
            g_num += gates.numel()
        return g_sum / g_num   
    
    def sparsities(self):
        sp_list = []
        for g in self.gating_modules:
            gates = g.gate.detach().cpu()
            sp_list.append(gates.sum() / gates.numel())
        return sp_list
    
    #############---contrastive loss functions---#############
    def initialize_contrast_settings(self, contrast_block_index, ndata, nn_num=5, nn_epoch=20):
        '''Initialize the setting for the contrastive loss.

           contrast_block_index: indicate which blocks need contrastive loss.
           ndata: length of dataset.
        '''
        self.contrast_blocks = []
        print(contrast_block_index)
        
        i = 0
        for m in self.modules():
            if isinstance(m, BasicBlock):
                if i in contrast_block_index: 
                    self.contrast_blocks.append(m)
                    m.use_contrast = True
                    m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    
                    ## memory banks and loss.
                    ndim = m.gate.n_dim
                    m.gate.gate_mb = LinearAverage(ndim, ndata)
                    m.gate.feat_mb = FeatureMemoryBank(ndim, ndata, nn_num, nn_epoch)
                    m.gate.criterion = KNNCriterion(nn_num, nn_epoch)
                i += 1        
        return None
    
    def get_loss_ct(self, indexes, epoch):
        '''Contrastive loss of gates in the network.
        '''
        loss = 0.
        for b in self.contrast_blocks:
            loss += b.gate._loss_ct(b.avgfeat.detach(), indexes, epoch)
        return loss
    
def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)

def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)

def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)
