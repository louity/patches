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
if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'

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
        print(f'eingenvalues for patch size {patch_size}')
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
        trainloader_whitening = torch.utils.data.DataLoader(trainset_whitening, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
    elif args.dataset in ['imagenet32', 'imagenet64', 'imagenet128']:
        trainset_whitening = Imagenet32(args.path_train, transform=transforms.ToTensor(), sz=spatial_size, n_arrays=n_arrays_train)
        trainloader_whitening = torch.utils.data.DataLoader(
        trainset_whitening, batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


    return trainset, testset, trainset_whitening, trainloader, testloader, trainloader_whitening


def compute_K_nn(trainloader, net, K_nn, device, seed=0, with_patches=True):
    num_centers, n_channels = net.conv_weight.shape[0], net.conv_weight.shape[1]
    N = 0
    K_nn_dist = 10000.*torch.ones([num_centers, K_nn]).half().to(device)
    if with_patches:
        K_nns = torch.ones([n_channels, num_centers * K_nn]).half().to(device)
        list_indices = torch.range(0,num_centers*K_nn-1).view([num_centers,K_nn]).long()

    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs, _ = inputs.to(device), _.to(device)
        if with_patches:
            dist, out = net(inputs)
        else:
            dist = net(inputs)
        tmp_dist = torch.cat([K_nn_dist, dist], dim=1 )
        K_nn_dist, indicies = torch.topk(tmp_dist, K_nn, dim=1, largest=False)
        if with_patches:
            mask = indicies>=K_nn 
            selected_indicies = indicies[mask]-K_nn
            K_nns[:,list_indices[mask]] = out[:,selected_indicies]
            K_nns[:,list_indices[~mask]] = K_nns[:,indicies[~mask]]
    
    if with_patches:            
        return K_nn_dist, K_nns.view(-1,num_centers,K_nn)
    else:
        return K_nn_dist, 'no_patches'

def intrinsic_dim(K_nn_dist, K_min, device='cuda' ):
    K_nn_dist = K_nn_dist.double()
    num_centers, K_nn = K_nn_dist.shape
    counts = torch.range(K_min,K_nn-2).to(device)
    K_nn_dist = K_nn_dist[:,1:]
    mask = torch.sum((K_nn_dist==0), dim=1)==0
    K_nn_dist = K_nn_dist[mask,:]
    log_K_nn_dist = torch.log(K_nn_dist)

    cumsum = torch.cumsum( log_K_nn_dist, dim=1)
    Z = log_K_nn_dist[:,K_min:] - torch.einsum('mk,k->mk', cumsum[:,K_min-1:-1], 1./counts)
    mask = torch.sum(Z<=0, dim=1)==0
    Z = Z[mask,:]
    dim =   2./(Z)
    return dim

def build_network(trainloader_whitening, dataset,  n_channel_convolution,spatialsize_convolution, whitening_reg,normalize, numpy_seed, device='cuda', with_patches=True, net_type = 'Net', space_basis = False):
    if dataset =='cifar10':
        stride = 1
    else:
        stride = 2

    patches_mean, whitening_eigvecs, whitening_eigvals = compute_whitening_from_loader(trainloader_whitening, patch_size=spatialsize_convolution, stride=stride)
    all_images = trainloader_whitening.dataset.data
    spatial_size = all_images.shape[2]
    n_patches_hw = spatial_size - spatialsize_convolution + 1
    if whitening_reg >=0:
        inv_sqrt_eigvals = np.diag(1. / np.sqrt(whitening_eigvals + whitening_reg))
        whitening_op = whitening_eigvecs.dot(inv_sqrt_eigvals).astype('float32')
        real_whitening_op = whitening_eigvecs.dot(inv_sqrt_eigvals).dot(whitening_eigvecs.T).astype('float32')
        #whitening_op = whitening_op.dot(whitening_eigvecs.T)
        #whitening_op = 1.*whitening_eigvecs.astype('float32')
    else:
        whitening_op = np.eye(whitening_eigvals.size, dtype='float32')

    patches = select_patches_randomly(all_images, patch_size=spatialsize_convolution, n_patches=n_channel_convolution, seed=numpy_seed)
    #patches = all_images[:n_channel_convolution,:spatialsize_convolution,:spatialsize_convolution,:].transpose(0,3,1,2)
    patches = patches.astype('float64')
    patches /= 255.0
    orig_shape = patches.shape
    
    old_patches = 1.*patches
    patches = patches.reshape(patches.shape[0], -1) 
    #patches = patches - np.mean(patches, axis=1)[:,np.newaxis]
    patches = patches - patches_mean.reshape(1, -1)


    #whitened_patches = patches.reshape(patches.shape[0], -1) - patches_mean.reshape(1, -1)

    if space_basis:
        whitened_patches = (patches).dot(real_whitening_op).astype('float32')
    else:
        whitened_patches = (patches).dot(whitening_op).astype('float32')

    if normalize:
        patch_norm =  np.linalg.norm(whitened_patches, axis=1)
        #min_divisor = 1e-8
        #patch_norm[patch_norm<min_divisor]=1

        normalized_whitened_patches = (
                whitened_patches / np.expand_dims(patch_norm, axis=1)
            ).reshape(orig_shape)
        print(f'patches normalized whitened : {normalized_whitened_patches.shape}')
    else:
        normalized_whitened_patches = whitened_patches.reshape(orig_shape)



    minus_whitened_patches_mean = -torch.from_numpy(patches_mean.dot(whitening_op))
    whitening_operator = torch.from_numpy(whitening_op.T).view(whitening_op.shape[0], whitening_op.shape[1], 1, 1)

    kernel_convolution = torch.from_numpy(normalized_whitened_patches).view(n_channel_convolution, -1, 1, 1)
    


    #kernel_convolution = torch.from_numpy(patches).reshape(patches.shape[0],-1).unsqueeze(-1).unsqueeze(-1).float()
   #if torch.cuda.is_available():
    #    kernel_convolution = kernel_convolution.half()

    kernel_convolution = kernel_convolution.to(device)
    whitening_operator = whitening_operator.to(device)
    minus_whitened_patches_mean = minus_whitened_patches_mean.to(device)
    if net_type =='Net':
        net = Net(kernel_convolution, whitening_operator, minus_whitened_patches_mean, n_patches_hw, spatialsize_convolution, normalize, with_patches=with_patches).to(device)
    elif net_type=='spatialNet':
        net = spatialNet(kernel_convolution, whitening_operator, minus_whitened_patches_mean, n_patches_hw, spatialsize_convolution, normalize, with_patches=with_patches).to(device)
    return net, whitening_eigvals, old_patches,real_whitening_op



def compute_images_and_whitening_from_loader(loader, patch_size, seed=0, stride=1, device='cuda'):
    mean, covariance = None, None

    # compute the mean
    N = 0

    N_tot, N_h,N_w, N_c = loader.dataset.data.shape

    all_images = np.zeros([N_tot, N_h,N_w,N_c])
    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs, _ = inputs.to(device), _.to(device)
        patches = F.unfold(inputs, patch_size, stride=stride).transpose(0, 1).contiguous()
        patches = patches.view(patches.size(0), -1)
        n = inputs.size(0)
        batch_mean = patches.mean(dim=1, keepdims=True).double()
        all_images[N:N+n] = inputs.cpu().numpy().transpose(0,2,3,1)
        if mean is None:
            mean = batch_mean
        else:
            mean = N/(N+n)*mean + n/(N+n)*batch_mean
        N += n
    mean = mean.float()
    # compute the covariance
    N = 0
    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs, _ = inputs.to(device), _.to(device)
        patches = F.unfold(inputs, patch_size, stride=stride).transpose(0, 1).contiguous()
        patches = patches.view(patches.size(0), -1) - mean
        n = inputs.size(0)
        batch_covariance = (patches @ patches.t() / patches.size(1))
        if covariance is None:
            covariance = batch_covariance.double()
        else:
            covariance = N/(N+n)*covariance + n/(N+n)*batch_covariance.double()
        N += n

    (eigvals, eigvecs) = scipy.linalg.eigh(covariance.cpu().numpy())

    mean = mean.view(-1).cpu().numpy().astype('float32')

    return (mean, eigvecs, eigvals, all_images)




class Net(nn.Module):
    def __init__(self, conv_weight, whitening_operator, minus_whitened_patches_mean, n_patches_hw, spatialsize_convolution, normalize, with_patches=True):
        super(Net, self).__init__()
        self.conv_size = spatialsize_convolution
        self.n_patches_hw = n_patches_hw
        self.conv_weight = conv_weight
        self.whitening_operator = whitening_operator
        self.minus_whitened_patches_mean = minus_whitened_patches_mean
        self.eps = 1e-16
        self.normalize = normalize
        self.with_patches = with_patches
        self.norm2_patches = torch.sum(conv_weight**2, dim=1).unsqueeze(0)
    def forward(self, x, ):
        out = F.unfold(x, self.conv_size) # extract img patches
        out = out.view(out.size(0), out.size(1), self.n_patches_hw, self.n_patches_hw)
        out = F.conv2d(out, self.whitening_operator, self.minus_whitened_patches_mean, stride=2) # retreive mean and apply whitening
        if self.normalize:
            out = out / (out.norm(p=2, dim=1, keepdim=True)) # normalize
        #out = out.half()
        dist = F.conv2d(out, self.conv_weight)
        if self.normalize:
            dist = 2.*(1.- dist)
        else:
            dist = torch.sum(out**2, dim=1).unsqueeze(1) -2.*dist + self.norm2_patches
        dist = F.relu(dist)
        N_patches = dist.shape[1]
        N_channels = out.shape[1]
        dist = dist.transpose(0,1).reshape(N_patches, -1)
        if self.with_patches:
            out  = out.transpose(0,1).reshape(N_channels,-1)
            return dist.half(), out.half()
        else:
            return dist.half()

class spatialNet(nn.Module):
    def __init__(self, conv_weight, whitening_operator, minus_whitened_patches_mean, n_patches_hw, spatialsize_convolution, normalize, stride = 2, with_patches=True):
        super(spatialNet, self).__init__()
        self.conv_size = spatialsize_convolution
        self.n_patches_hw = n_patches_hw
        self.conv_weight = conv_weight
        self.whitening_operator = whitening_operator
        self.minus_whitened_patches_mean = minus_whitened_patches_mean
        self.eps = 1e-16
        self.normalize = normalize
        self.with_patches = with_patches
        self.norm2_patches = torch.sum(conv_weight**2, dim=1).unsqueeze(0)
        self.stride = 1
        self.padding = 4
    def forward(self, x, ):
        out = F.unfold(x, self.conv_size, stride = self.stride, padding=self.padding) # extract img patches
        n_patches_hw = int(np.sqrt(out.size(-1)))
        out = out.view(out.size(0), out.size(1), n_patches_hw , n_patches_hw)
        dist = F.conv2d(out, self.conv_weight)
        dist = torch.sum(out**2, dim=1).unsqueeze(1) -2.*dist + self.norm2_patches
        dist = F.relu(dist)
        #dist = dist.transpose(0,1)
        return dist


def compute_K_nn_patches(trainloader, net, K_nn, device, seed=0, b_size = 512):
    num_centers, n_channels = net.conv_weight.shape[0], net.conv_weight.shape[1]
    K_nn = min(num_centers, K_nn)
    N = 0
    K_nn_dist = None
    K_nns = None
    patch_size = net.conv_size
    
    patches = 1.*net.conv_weight
    ind_patches = torch.range(0,num_centers-1).long().to(device)
    patches = patches[:,:,0,0]
    patches  = torch.split(patches,b_size,dim=0)

    M = 0
    for m ,patch in enumerate(patches):
        b = patch.size(0)
        patch_index = 1*ind_patches[M:M+b]
        patch = patch.reshape(b,3,patch_size,patch_size)
        M += b
        dist = net(patch)
        if K_nn_dist is None:
            _, _, N_h, N_w = dist.shape
            K_nn_dist = 100000*torch.ones([K_nn,num_centers, N_h,N_w]).to(device)     
            list_indices = torch.range(0,num_centers*N_h*N_w*K_nn-1).view([K_nn,num_centers, N_h, N_w]).long().to(device)
            K_nns = -1*torch.ones([ num_centers * K_nn*N_h*N_w]).long().to(device)
            loc_indicies = torch.range(0,N_h*N_w*num_centers-1).view([num_centers, N_h, N_w]).long().to(device)
            loc_indicies = loc_indicies.repeat(K_nn, 1, 1, 1)
        
        tmp_dist = torch.cat([K_nn_dist, dist], dim=0 )
        K_nn_dist, indicies = torch.topk(tmp_dist, K_nn, dim=0, largest=False)  
        mask = indicies>=K_nn 
        selected_indicies = indicies[mask]-K_nn
        K_nns[list_indices[mask]] = patch_index[selected_indicies]
        K_nns[list_indices[~mask]] =  K_nns[ indicies[~mask]*K_nn +loc_indicies[~mask]]   
    return K_nn_dist, K_nns.view(K_nn,num_centers, N_h, N_w)


def compute_topological_order( start_patch , K_nns,  M, stride=2, NN=0, start_x=0, start_y=0):
    indices = torch.ones([M,M]).long()
    cur_patch = start_patch
    done = False
    CI = int(K_nns.shape[3]/2)+1
    cur_x_ind = int(M/2)
    cur_y_ind = int(M/2)
    indices[cur_x_ind,cur_y_ind] = cur_patch
    cur_raw_ind = cur_x_ind*M+cur_y_ind
    neighbors_x = np.array([-1,1,0])
    neighbors_y = np.array([-1,1,0])
    cur_set = set([cur_raw_ind])
    done_set = set([])
    used_patches = [cur_patch]
    while not done:
        cur_raw_ind = cur_set.pop()
        cur_y_ind = np.mod(cur_raw_ind,M) 
        cur_x_ind = int((cur_raw_ind - cur_y_ind)/M)
        successors_x = neighbors_x + cur_x_ind
        successors_y = neighbors_y + cur_y_ind
        successors_x = successors_x[  (successors_x>=0) * (successors_x<M)]
        successors_y = successors_y[  (successors_y>=0) * (successors_y<M)]
        successors = successors_x.reshape(1,-1)*M +  successors_y.reshape(-1,1)
        successors = np.reshape(successors,-1)

        successors = set(successors)
        successors = successors.difference(done_set)
        successors = successors.difference(cur_set)                
        successors = successors.difference(set([cur_raw_ind])) 
        suc_indicies = np.array(list(successors))
        cur_patch = indices[cur_x_ind,cur_y_ind]
        for suc_index in suc_indicies: 
            y_shift = np.mod(suc_index,M)
            x_shift = int((suc_index - y_shift)/M)
            y_shift = y_shift-cur_y_ind
            x_shift = x_shift-cur_x_ind

            done_used = False 
            NN_order = NN
            while not done_used:
                patch = K_nns[NN_order,cur_patch, CI +x_shift*stride, CI +y_shift*stride].item()
                if patch in used_patches and  NN_order<K_nns.shape[0]-1:
                    NN_order +=1
                else:
                    done_used = True

            indices[cur_x_ind+x_shift,cur_y_ind+y_shift] = patch
            used_patches.append(patch)
            #print(K_nns[NN,cur_patch,:,:])
        cur_set.update(successors)
        done_set.update([cur_raw_ind])
    
        done = (len(done_set)==M*M)
    return indices


def build_topographical_image(patches, indices):
    M,N = indices.shape
    K_size = patches.shape[2]
    ws = 0
    image = np.zeros([3,(M+ws)*K_size,(M+ws)*K_size])
    for i in range(M):
        for j in range(N):
            start_x = (i+ws)*K_size
            end_x = (i+1+ws)*K_size
            start_y = (j+ws)*K_size
            end_y = (j+1+ws)*K_size
            image[:,start_x:end_x,start_y:end_y] = patches[indices[i,j]]
    return image



