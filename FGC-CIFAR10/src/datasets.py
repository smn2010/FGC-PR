import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from PIL import Image
class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        ## return the index of instances for maintaining the Memory Bank.
        return img, target, index

def cifar10_instance_dataloader(args):
    """
        Dataloader for cifar10 dataset
    """
    normalize =  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])
    
    ## train dataloader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])

    train_dataset = CIFAR10Instance(args.data, train=True, 
                                    transform=transform_train, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True)
    
    ## test dataloader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = datasets.CIFAR10(args.data, train=False, 
                                   transform=transform_test, download=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, test_loader


def cifar10_dataloader(args):
    """
        Dataloader for cifar10 dataset
    """
    normalize =  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])
    
    ## train dataloader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,])

    train_dataset = datasets.CIFAR10(args.data, train=True, 
                                    transform=transform_train, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=True)
    
    ## test dataloader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = datasets.CIFAR10(args.data, train=False, 
                                   transform=transform_test, download=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True)
    
    return train_loader, test_loader