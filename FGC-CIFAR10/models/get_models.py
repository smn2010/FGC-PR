from .resnet_cifar10_fgc import resnet20, resnet32, resnet56
from .resnet_cifar10_fgc import BasicBlock

def get_res_for_cifar10(args, num_classes=10):
    
    if args.arch == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif args.arch == 'resnet32':
        model = resnet32(num_classes=num_classes)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=num_classes)
    else:
        raise ValueError
    return model

def get_res_block(args):
    if args.arch == 'resnet20' or args.arch == 'resnet32' or args.arch == 'resnet56':
        return BasicBlock
    else:
        raise ValueError
