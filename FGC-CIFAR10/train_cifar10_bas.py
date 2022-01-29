#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from src.utils import ProgressMeter, AverageMeter
from src.datasets import cifar10_dataloader
from models.get_models import get_res_for_cifar10

model_names = ['resnet20', 'resnet32', 'resnet56']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 BAS Training')
parser.add_argument('-data', metavar='DIR', default='cifar10', help='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')

################## optimzer parameters ##################
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[300, 375, 450], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--gamma', default=0.1, type=float, metavar='G',
                    help='gamma of SGD solver')

################## gpu parameters ##################
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

################## gpu parameters ##################
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

################## resume parameters ##################
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

################## batch-shaping parameters. ##################
parser.add_argument('--rho', type=float, default=0, help='l0 loss scalar parameter')
parser.add_argument('--l0_start', type=int, default=100, help='l0 loss start epoch')
parser.add_argument('--l0_end', type=int, default=300, help='l0 loss end epoch')

parser.add_argument('--bs_end', type=int, default=100, help='batch-shaping loss end epoch')
parser.add_argument('--zeta', type=float, default=0.75, help='batch-shaping loss scalar parameter')

################## log file parameters. ####################
parser.add_argument('--desp', type=str, help='additional description')
parser.add_argument('--log', type=str, default='logs', help='log directory')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq',  default=20, type=int, 
                    metavar='N', help='save frequency (default: 20)')

best_acc1 = 0

def main():
    global best_acc1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

##################### Data loading code #######################
    train_loader, val_loader = cifar10_dataloader(args)
    
    num_classes = len(train_loader.dataset.classes)
    print("=> number of classes {}".format(num_classes))
###############################################################

######################## create model #############################
    print("=> creating model '{}'".format(args.arch))
    model = get_res_for_cifar10(args, num_classes)
    
    print(model)
##############################################################    

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

##############################################################
# define loss function (criterion) and optimizer and scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    gate_params, base_params = model.get_params()
    optimizer = torch.optim.SGD(
        [{'params': gate_params, 'initial_lr': args.lr, 'weight_decay': 0},
        {'params': base_params, 'initial_lr': args.lr}], 
        lr=args.lr, momentum=args.momentum, 
        weight_decay=args.weight_decay, 
        nesterov=True)

##############################################################
# optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
                
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))        
##############################################################

################ set scheduler according to start_epoch ######

    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule, 
        gamma=args.gamma, 
        last_epoch=args.start_epoch-1)

##############################################################          
            
    cudnn.benchmark = True

###############################################################

    # create checkpoint folder to save model and logs
    time_now = (time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime()))
    checkpoint_path = os.path.join(args.log, '{}_{}'.format(time_now, args.desp))
    
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
    
    log_path = os.path.join(checkpoint_path, '{}.txt'.format(time_now))
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    
###############################################################
    
    # open log file
    f_log = open(log_path, 'w')
    log_all(f_log, optimizer, checkpoint_path, log_path, args)
    
##################### training code ###########################
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, f_log)
        
        # adjust_learning_rate
        train_scheduler.step()
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, f_log)
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        
        if epoch > args.schedule[1] and is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch':  args.arch,
                'best_acc1': best_acc1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=True, 
               filename=checkpoint_path.format(net=args.arch, epoch=epoch, type='best'))
            continue

        if not (epoch+1) % args.save_freq:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch':  args.arch,
                'best_acc1': best_acc1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, 
               filename=checkpoint_path.format(net=args.arch, epoch=epoch, type='regular'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
###############################################################


########################### train #############################
def train(train_loader, model, criterion, optimizer, epoch, args, f_log):
    
    if epoch < args.bs_end:
        zeta = args.zeta * (args.bs_end - epoch) / args.bs_end
    else:
        zeta = 0.
    
    if epoch < args.l0_start:
        rho = 0.
    elif epoch < args.l0_end:
        rho = (epoch + 1 - args.l0_start) * args.rho / (args.l0_end - args.l0_start)
    else:
        rho = args.rho    
    
    print('Training Epoch: {epoch} Lr: {lr:.2e} Zeta: {zeta:.4f} '
          'Rho: {rho:.4f}'.format(epoch=epoch, 
                                  lr=optimizer.param_groups[0]['lr'], 
                                  zeta=zeta, rho=rho))

    if f_log is not None:
        print('Training Epoch: {epoch} Lr: {lr:.2e} Zeta: {zeta:.4f} '
          'Rho: {rho:.4f}'.format(epoch=epoch, 
                                  lr=optimizer.param_groups[0]['lr'], 
                                  zeta=zeta, rho=rho), file=f_log)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc', ':6.2f')
    train_loss_cls = AverageMeter('Loss_cls', ':.4f')
    train_loss_bs  = AverageMeter('Loss_bs', ':.4f')
    train_loss_l0  = AverageMeter('Loss_l0', ':.4f')
    train_sparsity = AverageMeter('Sparsity', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, train_loss_cls, train_loss_bs, train_loss_l0, train_sparsity],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(images)
        loss_cls = criterion(output, target)
        
        ## L0 regularization loss
        if rho == 0:
            loss_l0 = torch.tensor([0.]).to(args.gpu)
        else:
            loss_l0 = rho * model.get_loss_l0()
                
        ## batch-shaping loss
        if zeta == 0:
            loss_bs = torch.tensor([0.]).cuda(args.gpu)
        else:
            loss_bs = zeta * model.get_loss_bs()
        
        loss = loss_cls + loss_bs + loss_l0
        
        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        #print(acc1, acc1[0], loss_cls.item())
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        train_loss_cls.update(loss_cls.item(), images.size(0))
        train_loss_bs.update(loss_bs.item(), images.size(0))
        train_loss_l0.update(loss_l0.item(), images.size(0))
        train_sparsity.update(model.sparsity(), images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i, f_log)
        
###############################################################


########################## validate ###########################
def validate(val_loader, model, criterion, args, f_log):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    sparsity = AverageMeter('Sparsity', ':.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, sparsity],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            sparsity.update(model.sparsity(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, f_log)
        
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1), file=f_log)
        
    return top1.avg
        
###############################################################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')


def log_all(f_log, optimizer, checkpoint_path, log_path, args):
    print('train_cifar10_bas.py\n', file=f_log)

    print("Optimizer: {}".format(optimizer), file=f_log)

    print("checkpoint_path: {}\nlog_path: {}\n".
         format(checkpoint_path, log_path), file=f_log)

    print("Args: {}\n".format(args), file=f_log)
    #print("GPU devices: {}\n".format(os.environ["CUDA_VISIBLE_DEVICES"]), file=f_log)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
