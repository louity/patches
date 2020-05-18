import argparse
import numpy as np

import torch 
import torch.nn as nn
import torchvision
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from utils import *
from torchvision.datasets import CIFAR10

from imagenet import Imagenet32



def pca_patches(trainset,trainloader_whitening,patch_size,whitening_reg):
    t = trainset.data
    #patches = select_patches_randomly(t,patch_size)
    (mean, eigvecs, eigvals) = compute_whitening_from_loader(trainloader_whitening, patch_size)

    whitened_eigvals = eigvals*np.power(eigvals + whitening_reg, -1./2)



def eigvals_patches(trainloader_whitening,patch_sizes,whitening_regs):
    whitening_regs = torch.from_numpy(whitening_regs).unsqueeze(0)
    all_whitened_eigvals= []
    for patch_size in patch_sizes:
        (mean, eigvecs, eigvals) = compute_whitening_from_loader(trainloader_whitening, patch_size)
        eigvals = 1.*eigvals[::-1]
        eigvals = torch.from_numpy(eigvals)
        whitening = torch.pow(eigvals.unsqueeze(1)+ whitening_regs, -1)
        whitening_2 = torch.pow(eigvals.unsqueeze(1)+ whitening_regs, -2)
        whitened_eigvals = torch.einsum( 'i,ik->ik', eigvals,whitening)
        whitened_eigvals_2 = torch.einsum( 'i,ik->ik', eigvals,whitening_2)
        whitened_eigvals = torch.cat([eigvals.unsqueeze(1), whitened_eigvals,whitened_eigvals_2], dim=1)
        whitened_eigvals = torch.einsum('ij,j->ij',whitened_eigvals,1./whitened_eigvals[0,:])
        all_whitened_eigvals.append(whitened_eigvals)

    #all_whitened_eigvals = torch.stack(all_whitened_eigvals)
    #all_whitened_eigvals = torch.einsum('ijk,ik->ijk',all_whitened_eigvals,1./all_whitened_eigvals[:,0,:])

    return all_whitened_eigvals


def cov_dimension(eigvals, thresh, cum_sum_eigvals=None):
    if cum_sum_eigvals is None:
        cum_sum_eigvals = np.cumsum(eigvals, axis = 0)
        cum_sum_eigvals = np.einsum('ij,j->ij', cum_sum_eigvals, 1./cum_sum_eigvals[-1,:])

    test = 1.*(cum_sum_eigvals > (1-thresh))
    cov_dim = (test.shape[0] - np.sum(test, axis=0))/test.shape[0]
    return cov_dim






def load_data(args):
    if args.dataset == 'cifar10':
        spatial_size = 32
        padding = 0 if args.no_padding else 4
        transform_train = transforms.Compose([
            transforms.RandomCrop(spatial_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
        n_classes=10


        testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
    elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        n_arrays_train = 10
        padding = 4
        spatial_size = 32
        if args.dataset=='imagenet64':
            spatial_size = 64
            padding = 8
        if args.dataset=='imagenet128':
            spatial_size = 128
            padding = 16
            n_arrays_train = 100
        n_classes = 1000

        if args.no_padding:
            padding = 0

        transforms_train = [
            transforms.RandomCrop(spatial_size, padding=padding, padding_mode=args.padding_mode),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        transforms_test = [transforms.ToTensor(), normalize]

        trainset = Imagenet32(args.path_train, transform=transforms.Compose(transforms_train), sz=spatial_size, n_arrays=n_arrays_train)
        testset = Imagenet32(args.path_test, transform=transforms.Compose(transforms_test), sz=spatial_size)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batchsize, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batchsize, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        n_classes = 1000

    if args.dataset == 'cifar10':
        trainset_whitening = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        trainloader_whitening = torch.utils.data.DataLoader(trainset_whitening, batch_size=1000, shuffle=False, num_workers=args.num_workers)
    elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:
        trainset_whitening = Imagenet32(args.path_train, transform=transforms.ToTensor(), sz=spatial_size, n_arrays=n_arrays_train)
        trainloader_whitening = torch.utils.data.DataLoader(
        trainset_whitening, batch_size=500, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


    return trainset, testset, trainset_whitening, trainloader, testloader, trainloader_whitening











