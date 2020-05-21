import argparse
import numpy as np
import seaborn as sns

import torch 
import torch.nn as nn
import torchvision
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import analyze_patches as pa 

import os


# arguments
class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)
args = {'dataset':'cifar10',
		'no_padding':False,
		'batchsize':128,
		'num_workers':4,
		'padding_mode':'constant',
		'path_train': '/nfs/gatsbystor/michaela/projects/data/imagenet/imagenet32/out_data_train/',
		'path_test': '/nfs/gatsbystor/michaela/projects/data/imagenet/imagenet32/out_data_val/',
		'path_save':'/nfs/gatsbystor/michaela/projects/nondeep/data/',
		'whitening_reg':0.001,
		'K_nn':3,
		'numpy_seed':0,
		'n_channel_convolution':32000,
		'K_min':10,
		'normalize':True,
		'device':'cuda',
		'with_patches':False,
		'M':10,
		'min_divisor':1e-8,
		'space_basis':True,
		're_whiten':False
	   }
args = Struct(**args)

patch_sizes = [20,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
patch_sizes = [6]

whitening_regs = np.array([0.00001,0.0001,0.001,0.01,0.1,1.,10.])

# Getting data loaders
trainset, testset, trainset_whitening, trainloader, testloader, trainloader_whitening = pa.load_data(args)
for patch_size in patch_sizes:
	print(f'Computing K_nn for patch size {patch_size}')
	net,whitening_eigvals, old_patches,whitening_eigvecs = pa.build_network(trainloader_whitening, args.dataset, args.n_channel_convolution, patch_size , args.whitening_reg, args.normalize, args.numpy_seed,net_type = 'spatialNet', space_basis=args.space_basis)
	K_nn_dist, K_nns = pa.compute_K_nn_patches(trainloader_whitening,net,args.K_nn, args.device, b_size = args.batchsize)
	k_nn_file = args.path_save + 'patches_k_nns_'+str(args.dataset)+'_whitening_reg_'+ str(args.whitening_reg)+'_patchsize_'+str(patch_size)+'.npz'
	patches = net.conv_weight
	indicies = pa.compute_topological_order(0,K_nns,args.M)
	K_nn_dist = K_nn_dist.cpu().numpy()
	K_nns = K_nns.cpu().numpy()
	patches = patches[:,:,0,0]

	if args.re_whiten:

		whitening_eigvecs  = torch.from_numpy(whitening_eigvecs).to(args.device).float()
		patches = torch.einsum('nk,lk->nl',patches,whitening_eigvecs)

		patch_norm =  torch.norm(patches, dim=1)

		patches = patches / patch_norm.unsqueeze(-1)


	patches = patches.reshape(patches.size(0), 3, patch_size,patch_size)
	patches = patches.cpu().numpy()
	

	#patches = old_patches
	
	image = pa.build_topographical_image(patches, indicies)

	np.savez(k_nn_file, K_nn_dist=K_nn_dist,
			K_nns=K_nns,  patches= patches, image=image)
	print(f'K_nn computed and savec in file {k_nn_file}')







