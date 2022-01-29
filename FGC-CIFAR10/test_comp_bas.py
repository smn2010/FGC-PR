#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset of a model
print pruning ratio and computation.
"""
import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from src.datasets import cifar10_dataloader
from models.get_models import get_res_for_cifar10, get_res_block

GATE_MAC = []
def hook_gate_conv_layer(module, input, output):
    Ci = module.in_channels
    Co = module.out_channels
    K1, K2  = module.kernel_size
    H, W = output.size()[2:4]
    
    MAC  = Ci * Co * H * W * K1 * K2 
    GATE_MAC.append(MAC)

BLOCK_MAC = []
def hook_block_conv_layer(module, input, output):
    Ci = module.in_channels
    Co = module.out_channels
    K1, K2  = module.kernel_size
    H, W = output.size()[2:4]

    MAC = Ci * Co * H * W * K1 * K2 
    BLOCK_MAC.append(MAC)


model_names = ['resnet20', 'resnet32', 'resnet56']

parser = argparse.ArgumentParser(description='PyTorch FGC Testing')
parser.add_argument('-data', metavar='DIR', default='imagenet', help='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

    
parser.add_argument('--gpu', type=int, required=True, help='GPU id to use.') 
parser.add_argument('--resume', type=str, required=True, help='the weights of model you want to test')
parser.add_argument('--log', type=str, default='./results/results.txt', help='log file')

    

def main():
    args = parser.parse_args()
        
    args.distributed = False
    
    ##################### Data loading code #######################
    train_loader, test_loader = cifar10_dataloader(args)
    num_classes = len(train_loader.dataset.classes)
    print("=> number of classes {}".format(num_classes))
    ###############################################################
    
    ######################## create model ########################
    print("=> creating model '{}'".format(args.arch))
    model = get_res_for_cifar10(args, num_classes)    
    BLOCK = get_res_block(args)
    #print(model)
    ##############################################################
  
    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Only one gpu is supported.")
    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}, strict=False)
            
            #model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    model.eval()
    
    ########################## register hooks ###################
    handles = []
    handles.append(model.conv1.register_forward_hook(hook_block_conv_layer))
    for block in model.modules():
        if isinstance(block, BLOCK):
            ##block.gate.register_forward_hook(hook_gate)
            handles.append(block.gate.mlp.fc1.register_forward_hook(hook_gate_conv_layer))
            handles.append(block.gate.mlp.fc2.register_forward_hook(hook_gate_conv_layer))
            for module in block.children():
                if isinstance(module, nn.Conv2d):
                    handles.append(module.register_forward_hook(hook_block_conv_layer))

    ########################## base computation ###################
    image, _ = iter(test_loader).next()
    model(image.cuda())
    base_gate_mac  = np.sum(GATE_MAC)
    base_block_mac = np.sum(BLOCK_MAC)
    #print(len(BLOCK_MAC), BLOCK_MAC)
    print("Base Gate MACs: {:d}; Base Block MACs: {:d}".format(base_gate_mac, base_block_mac))
    
    for handle in handles: handle.remove()    

    ########################## accuracy and MACs ###################
    correct_1 = 0.0
    correct_5 = 0.0
    total_block_mac = 0
    for n_iter, (image, label) in enumerate(test_loader):

        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top1 
        correct_1 += correct[:, :1].sum()
        
        sparsities = model.sparsities()
        for i, sp in enumerate(sparsities): 
            total_block_mac  += sp * (BLOCK_MAC[2*i+1] + BLOCK_MAC[2*i+2])
        total_block_mac += BLOCK_MAC[0]
        
    avg_block_mac = int(total_block_mac / (n_iter+1))
    avg_mac = int(avg_block_mac + base_gate_mac)
    
    print("Gated Block MACs: {:d}, Total MACs: {:d}".format(avg_block_mac, avg_mac))
    
    acc1 = (correct_1 / len(test_loader.dataset)).item()
    acc5 = (correct_5 / len(test_loader.dataset)).item()    

    print("@Acc1: {:.4f}; @Acc5: {:.4f}; MACs: {:d} ".format(acc1, acc5, avg_mac))
    pruning = 1-float(avg_mac)/base_block_mac
    print("@Err1: {:.4f}; @Err5: {:.4f}; Pruning: {:.4f}".format(1-acc1, 1-acc5, pruning))
    print("\n")
    
    f = open(args.log, 'a')
    print('{:.4f}  {:.4f}  {:d} {:.4f} {:.4f} {:.4f} {:s}'.
          format(acc1, acc5, avg_mac, 1-acc1, 1-acc5, pruning, args.resume), file=f)
    f.close()

    
if __name__ == '__main__':
    main()
