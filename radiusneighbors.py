import argparse
import ast
import hashlib
import json
import numpy as np
import os
import time

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from imagenet import Imagenet32
from utils import compute_whitening, compute_whitening_from_loader, correct_topk, select_patches_randomly, heaviside, topk, topk_heaviside, compute_classifier_outputs, create_classifier_blocks, compute_channel_mean_and_std

print('metric.py')
parser = argparse.ArgumentParser('classification on cifar10 based on patches neighborhood indicators for euclidian metric')
# parameters for the patches
parser.add_argument('--dataset', help="cifar10/?", default='cifar10')
parser.add_argument('--no_padding', action='store_true', help='no padding used')
parser.add_argument('--patches_file', help=".t7 file containing patches", default='')
parser.add_argument('--n_channel_convolution', default=256, type=int)
parser.add_argument('--spatialsize_convolution', default=6, type=int)
parser.add_argument('--padding_mode', default='constant', choices=['constant', 'reflect', 'symmetric'], help='type of padding for torch RandomCrop')
parser.add_argument('--whitening_reg', default=0.001, type=float, help='regularization bias for zca whitening, negative values means no whitening')

# parameters for the extraction
parser.add_argument('--stride_convolution', default=1, type=int)
parser.add_argument('--stride_avg_pooling', default=2, type=int)
parser.add_argument('--spatialsize_avg_pooling', default=5, type=int)
parser.add_argument('--bias', type=float, default=0.1)
parser.add_argument('--shrink', default='softshrink', choices=['softshrink', 'hardshrink', 'heaviside', 'topk', 'topk_heaviside', 'topk_heaviside_and_heaviside'], help='type of shrink function')
parser.add_argument('--topk_fraction', default=0.25, type=float)

# parameters of the classifier
parser.add_argument('--batch_norm', action='store_true', help='add batchnorm before classifier')
parser.add_argument('--no_affine_batch_norm', action='store_true', help='affine=False in batch norms')
parser.add_argument('--normalize_net_outputs', action='store_true', help='precompute the mean and std of the outputs to normalize them (alternative to batch norm)')
parser.add_argument('--bottleneck_dim', default=0, type=int, help='bottleneck dimension for the classifier')
parser.add_argument('--convolutional_classifier', type=int, default=0, help='size of the convolution for convolutional classifier')
parser.add_argument('--bottleneck_spatialsize', type=int, default=1, help='spatial size of the bottleneck')
parser.add_argument('--relu_after_bottleneck', action='store_true', help='add relu after bottleneck ')
parser.add_argument('--dropout', type=float, default=0., help='dropout after relu')
parser.add_argument('--feat_square', action='store_true', help='add square features')

# parameters of the optimizer
parser.add_argument('--batchsize', type=int, default=512)
parser.add_argument('--batchsize_net', type=int, default=0)
parser.add_argument('--lr_schedule', type=str, default='{0:1e-3, 1:1e-4}')
parser.add_argument('--nepochs', type=int, default=90)
parser.add_argument('--optimizer', choices=['Adam', 'SGD'], default='Adam')
parser.add_argument('--sgd_momentum', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=0.)

# hardware parameters
parser.add_argument('--path_train', help="path to imagenet", default='/d1/dataset/imagenet32/out_data_train')
parser.add_argument('--path_test', help="path to imagenet", default='/d1/dataset/imagenet32/out_data_val')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--no_cudnn', action='store_true', help='disable cuDNN to prevent cuDNN error (slower)')
parser.add_argument('--no_jit', action='store_true', help='disable torch.jit optimization to prevent error (slower)')

# reproducibility parameters
parser.add_argument('--numpy_seed', type=int, default=0)
parser.add_argument('--torch_seed', type=int, default=0)
parser.add_argument('--save_model', action='store_true', help='saves the model')
parser.add_argument('--save_best_model', action='store_true', help='saves the best model')
parser.add_argument('--resume', default='', help='filepath of checkpoint to load the model')

args = parser.parse_args()


if args.batchsize_net > 0:
    assert args.n_channel_convolution // args.batchsize_net == args.n_channel_convolution / args.batchsize_net, 'batchsize_net must divide n_channel_convolution'

print(f'Arguments : {args}')


learning_rates = ast.literal_eval(args.lr_schedule)

# Extract the parameters
n_channel_convolution = args.n_channel_convolution
stride_convolution = args.stride_convolution
spatialsize_convolution = args.spatialsize_convolution
stride_avg_pooling = args.stride_avg_pooling
spatialsize_avg_pooling = args.spatialsize_avg_pooling
if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'
print(f'device: {device}')
torch.manual_seed(args.torch_seed)
np.random.seed(args.numpy_seed)

train_sampler = None

# Define the dataset
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


class Net(nn.Module):
    def __init__(self, n_patches_hw, spatialsize_convolution, spatialsize_avg_pooling, stride_avg_pooling, shrink, bias=1.0):
        super(Net, self).__init__()
        self.pool_size = spatialsize_avg_pooling
        self.pool_stride = stride_avg_pooling
        self.bias = bias
        self.conv_size = spatialsize_convolution
        self.n_patches_hw = n_patches_hw
        if shrink == 'heaviside':
            self.shrink = heaviside
        elif shrink == 'topk':
            # self.bias = int((1 - args.topk_fraction) * n_channel_convolution)
            self.shrink = topk
        elif shrink == 'topk_heaviside':
            self.bias = int((1 - args.topk_fraction) * n_channel_convolution)
            self.shrink = topk_heaviside
        elif shrink == 'hardshrink':
            self.shrink = F.hardshrink
        elif shrink == 'softshrink':
            self.shrink = F.softshrink

    def forward(self, x, conv_weight, whitening_operator, minus_whitened_patches_mean):
        out = F.unfold(x, self.conv_size) # extract img patches
        out = out.view(out.size(0), out.size(1), self.n_patches_hw, self.n_patches_hw)
        out = F.conv2d(out, whitening_operator, minus_whitened_patches_mean) # retreive mean and apply whitening
        out = out / (out.norm(p=2, dim=1, keepdim=True)) # normalize
        out = out.half()
        out = F.conv2d(out, conv_weight)

        out1 = self.shrink(-out, self.bias)
        out1 = F.avg_pool2d(out1, [self.pool_size, self.pool_size], stride=[self.pool_stride, self.pool_stride],
                            ceil_mode=True)
        out2 = self.shrink(out, self.bias)
        out2 = F.avg_pool2d(out2, [self.pool_size, self.pool_size], stride=[self.pool_stride, self.pool_stride],
                            ceil_mode=True)
        return out1.float(), out2.float()


# # old version, whitening computed on a subset
# patches_mean, whitening_op  = compute_whitening(patches, reg=args.whitening_reg)

# new version, whitening computed on all the patches of the dataset
whitening_file = f'data/whitening_{args.dataset}_patchsize{spatialsize_convolution}.npz'

if not os.path.exists(whitening_file):
    if args.dataset == 'cifar10':
        trainset_whitening = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        trainloader_whitening = torch.utils.data.DataLoader(trainset_whitening, batch_size=1000, shuffle=False, num_workers=args.num_workers)
    elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:
        trainset_whitening = Imagenet32(args.path_train, transform=transforms.ToTensor(), sz=spatial_size, n_arrays=n_arrays_train)
        trainloader_whitening = torch.utils.data.DataLoader(
            trainset_whitening, batch_size=500, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    patches_mean, whitening_eigvecs, whitening_eigvals  = compute_whitening_from_loader(trainloader_whitening, patch_size=spatialsize_convolution)
    del trainloader_whitening
    del trainset_whitening
    np.savez(whitening_file, patches_mean=patches_mean,
             whitening_eigvecs=whitening_eigvecs,
             whitening_eigvals=whitening_eigvals)
    print(f'Whitening computed and saved in file {whitening_file}')

whitening = np.load(whitening_file)
whitening_eigvecs = whitening['whitening_eigvecs']
whitening_eigvals = whitening['whitening_eigvals']
patches_mean = whitening['patches_mean']

inv_sqrt_eigvals = np.diag(np.power(whitening_eigvals + args.whitening_reg, -1/2))
whitening_op = whitening_eigvecs.dot(inv_sqrt_eigvals).astype('float32')

t = trainset.data
print(f'Trainset : {t.shape}')

patches = select_patches_randomly(t, patch_size=spatialsize_convolution, n_patches=n_channel_convolution, seed=args.numpy_seed)
print(f'patches randomly selected: {patches.shape}')

patches = patches.astype('float64')
patches /= 255.0
orig_shape = patches.shape
patches = patches.reshape(patches.shape[0], -1) - patches_mean.reshape(1, -1)
whitened_patches = (patches).dot(whitening_op).astype('float32')

normalized_whitened_patches = (
        whitened_patches / np.linalg.norm(whitened_patches, axis=1, keepdims=True)
    ).reshape(orig_shape)
print(f'patches normalized whitened : {normalized_whitened_patches.shape}')

minus_whitened_patches_mean = -torch.from_numpy(patches_mean.dot(whitening_op))
whitening_operator = torch.from_numpy(whitening_op.T).view(whitening_op.shape[0], whitening_op.shape[1], 1, 1)

kernel_convolution = torch.from_numpy(normalized_whitened_patches).view(n_channel_convolution, -1, 1, 1)

params = []


if torch.cuda.is_available():
    kernel_convolution = kernel_convolution.half()

criterion = nn.CrossEntropyLoss()


n_patches_hw = spatial_size - spatialsize_convolution + 1
net = Net(n_patches_hw, spatialsize_convolution, spatialsize_avg_pooling,
          stride_avg_pooling, shrink=args.shrink, bias=args.bias).to(device)

kernel_convolution = kernel_convolution.to(device)
whitening_operator = whitening_operator.to(device)
minus_whitened_patches_mean = minus_whitened_patches_mean.to(device)

x = torch.rand(1, 3, spatial_size, spatial_size).to(device)


if args.no_cudnn:
    torch.backends.cudnn.enabled = False
else:
    cudnn.benchmark = True


out1, out2 = net(x, kernel_convolution, whitening_operator, minus_whitened_patches_mean)
print(f'Net output size: out1 {out1.shape[-3:]} out2 {out2.shape[-3:]}')

classifier_blocks = create_classifier_blocks(out1, out2, args, params, n_classes)

print(f'Parameters shape {[param.shape for param in params]}')
print(f'N parameters : {sum([np.prod(list(param.shape)) for param in params])/1e6} millions')

del x, out1, out2

if args.batchsize_net > 0:
    kernel_convolution = [kernel_convolution[i*args.batchsize_net:(i+1)*args.batchsize_net] for i in range(args.n_channel_convolution // args.batchsize_net)]
else:
    kernel_convolution = [kernel_convolution]

if torch.cuda.is_available() and not args.no_jit:
    print('optimizing net execution with torch.jit')
    trial = torch.rand(args.batchsize//n_gpus, 3, spatial_size, spatial_size).to(device)

    inputs = {'forward': (trial, kernel_convolution[0], whitening_operator, minus_whitened_patches_mean)}
    with torch.jit.optimized_execution(True):
        net = torch.jit.trace_module(net, inputs, check_trace=False, check_tolerance=False)
    del inputs
    del trial


if args.multigpu and n_gpus > 1:
    print(f'{n_gpus} available, using Dataparralel for net')
    net = nn.DataParallel(net)

if args.normalize_net_outputs:
    mean_std_file = f'data/mean_std_{args.dataset}_seed{args.numpy_seed}_patchsize{spatialsize_convolution}_npatches{args.n_channel_convolution}_reg{args.whitening_reg}_shrink{args.shrink}.npz'
    if not os.path.exists(mean_std_file):
        mean1, mean2, std1, std2 = compute_channel_mean_and_std(trainloader, net, n_channel_convolution,
            kernel_convolution, whitening_operator, minus_whitened_patches_mean, n_epochs=1, seed=0)
        np.savez(mean_std_file, mean1=mean1.cpu().numpy(), mean2=mean2.cpu().numpy(), std1=std1.cpu().numpy(), std2=std2.cpu().numpy())
        print(f'Net outputs mean and std computed and saved in file {mean_std_file}')
    mean_std = np.load(mean_std_file)
    mean1 = torch.from_numpy(mean_std['mean1']).to(device)
    mean2 = torch.from_numpy(mean_std['mean2']).to(device)
    std1 = torch.from_numpy(mean_std['std1']).to(device)
    std2 = torch.from_numpy(mean_std['std2']).to(device)


def train(epoch):
    net.train()
    batch_norm1, batch_norm2, classifier1, classifier2, classifier = classifier_blocks
    for bn in [batch_norm1, batch_norm2]:
        if bn is not None:
            bn.train()

    train_loss = 0
    total=0
    correct=0


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            if len(kernel_convolution) > 1:
                outputs = []
                for i in range(len(kernel_convolution)):
                    outputs.append(net(inputs, kernel_convolution[i], whitening_operator, minus_whitened_patches_mean))
                outputs1 = torch.cat([out[0] for out in outputs], dim=1)
                outputs2 = torch.cat([out[1] for out in outputs], dim=1)
                del outputs
            else:
                outputs1, outputs2 = net(inputs, kernel_convolution[0], whitening_operator, minus_whitened_patches_mean)

            if args.normalize_net_outputs:
                outputs1 = (outputs1 - mean1) / std1
                outputs2 = (outputs2 - mean2) / std2

        optimizer.zero_grad()
        outputs, targets = compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1,
                batch_norm2, classifier1, classifier2, classifier,
                train=True)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            return False
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train, epoch: {}; Loss: {:.2f} | Acc: {:.1f} ; threshold {:.3f}'.format(
        epoch, train_loss / (batch_idx + 1), 100. * correct / total, args.bias))
    return True


def test(epoch, loader=testloader, msg='Test'):
    global best_acc
    net.eval()
    batch_norm1, batch_norm2, classifier1, classifier2, classifier = classifier_blocks
    for bn in [batch_norm1, batch_norm2]:
        if bn is not None:
            bn.eval()

    test_loss = 0
    correct_top1, correct_top5 = 0, 0
    total = 0
    outputs_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if len(kernel_convolution) > 1:
                outputs = []
                for i in range(len(kernel_convolution)):
                    outputs.append(net(inputs, kernel_convolution[i], whitening_operator, minus_whitened_patches_mean))
                outputs1 = torch.cat([out[0] for out in outputs], dim=1)
                outputs2 = torch.cat([out[1] for out in outputs], dim=1)
                del outputs
            else:
                outputs1, outputs2 = net(inputs, kernel_convolution[0], whitening_operator, minus_whitened_patches_mean)

            if args.normalize_net_outputs:
                outputs1 = (outputs1 - mean1) / std1
                outputs2 = (outputs2 - mean2) / std2


            outputs, targets = compute_classifier_outputs(
                outputs1, outputs2, targets, args, batch_norm1,
                batch_norm2, classifier1, classifier2, classifier,
                train=True)
            loss = criterion(outputs, targets)

            outputs_list.append(outputs)

            test_loss += loss.item()
            cor_top1, cor_top5 = correct_topk(outputs, targets, topk=(1, 5))
            correct_top1 += cor_top1
            correct_top5 += cor_top5
            _, predicted = outputs.max(1)
            total += targets.size(0)

        test_loss /= (batch_idx + 1)
        acc1, acc5 = 100. * correct_top1 / total, 100. * correct_top5 / total

        print(f'{msg}, epoch: {epoch}; Loss: {test_loss:.2f} | Acc: {acc1:.1f} @1 {acc5:.1f} @5 ; threshold {args.bias:.3f}')

        outputs = torch.cat(outputs_list, dim=0).cpu()


        return acc1, outputs

hashname = hashlib.md5(str.encode(json.dumps(vars(args), sort_keys=True))).hexdigest()
if args.save_model:
    checkpoint_dir = f'checkpoints/{args.dataset}_{args.n_channel_convolution}patches_{args.spatialsize_convolution}x{args.spatialsize_convolution}/{args.optimizer}_{args.lr_schedule}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_file = os.path.join(checkpoint_dir, f'{hashname}.pth.tar')
    print(f'Model will be saved at file {checkpoint_file}.')

    state = {'args': args}
    if os.path.exists(checkpoint_file):
        state = torch.load(checkpoint_file)

start_epoch = 0
if args.resume:
    state = torch.load(args.resume)
    start_epoch = state['epoch'] + 1
    print(f'Resuming from file {args.resume}, start epoch {start_epoch}...')
    if start_epoch not in learning_rates:
        closest_i = max([i for i in learning_rates if i <= start_epoch])
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rates[closest_i], weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=learning_rates[closest_i], momentum=args.sgd_momentum, weight_decay=args.weight_decay)
        optimizer.load_state_dict(state['optimizer'])

    for block, name in zip(classifier_blocks, ['bn1','bn2','cl1', 'cl2', 'cl' ]):
        if block is not None:
            block.load_state_dict(state['name'])
    acc, outputs = test(-1)

start_time = time.time()
best_test_acc, best_epoch = 0, -1

for i in range(start_epoch, args.nepochs):
    if i in learning_rates:
        print('new lr:'+str(learning_rates[i]))
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rates[i], weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=learning_rates[i], momentum=args.sgd_momentum, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError('optimizer {} not implemented'.format(args.optimizer))
    no_nan_in_train_loss = train(i)
    if not no_nan_in_train_loss:
        print(f'Epoch {i}, nan in loss, stopping training')
        break
    test_acc, outputs = test(i)

    if test_acc > best_test_acc:
        print(f'Best acc ({test_acc}).')
        best_test_acc = test_acc
        best_epoch = i

    if args.save_model or args.save_best_model and best_epoch == i:
        print(f'saving...')
        state.update({
            'optimizer': optimizer.state_dict(),
            'epoch': i,
            'acc': test_acc,
            'outputs': outputs,
        })
        for block, name in zip(classifier_blocks, ['bn1','bn2','cl1', 'cl2', 'cl']):
            if block is not None:
                state.update({
                    name: block.state_dict()
                })
        torch.save(state, checkpoint_file)

print(f'Best test acc. {best_test_acc}  at epoch {best_epoch}/{i}')
hours = (time.time() - start_time) / 3600
print(f'Done in {hours:.1f} hours with {n_gpus} GPU')
