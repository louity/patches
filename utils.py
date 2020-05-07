import numpy as np
import torch
import scipy.linalg
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    device = 'cuda'
    n_gpus = torch.cuda.device_count()
else:
    device = 'cpu'


def select_patches_randomly(images, patch_size, n_patches=5000000, seed=0):
    np.random.seed(seed)
    images = images.transpose(0, 3, 1, 2)

    n_patches_per_row = images.shape[2] - patch_size + 1
    n_patches_per_col = images.shape[3] - patch_size + 1
    n_patches_per_image = n_patches_per_row * n_patches_per_col
    n_patches_total = images.shape[0] * n_patches_per_image
    patch_ids = np.random.choice(n_patches_total, size=n_patches, replace=True)
    unique_patch_ids = np.unique(patch_ids)
    while len(unique_patch_ids) < len(patch_ids):
        unique_patch_ids = np.unique(np.concatenate([unique_patch_ids, np.random.choice(n_patches_total, size=n_patches, replace=True)]))
    patch_ids = unique_patch_ids[:len(patch_ids)]

    patches = np.zeros((n_patches, 3, patch_size, patch_size), dtype=images.dtype)

    for i_patch, patch_id in enumerate(patch_ids):
        img_id = patch_id % images.shape[0]
        x_id = patch_id // images.shape[0] % n_patches_per_row
        y_id = patch_id // (images.shape[0] * n_patches_per_row)
        patches[i_patch] = images[img_id, :, x_id:x_id+patch_size, y_id:y_id+patch_size]

    return patches


def correct_topk(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum().item()
            res.append(correct_k)
    return res


def compute_whitening_from_loader(loader, patch_size, seed=0, reg=1e-4):
    mean, covariance = None, None

    # compute the mean
    N = 0
    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs, _ = inputs.to(device), _.to(device)
        patches = F.unfold(inputs, patch_size).transpose(0, 1).contiguous()
        patches = patches.view(patches.size(0), -1)
        n = inputs.size(0)
        batch_mean = patches.mean(dim=1, keepdims=True).double()
        if mean is None:
            mean = batch_mean
        else:
            mean = N/(N+n)*mean + n/(N+n)*batch_mean
        N += n

    # compute the covariance
    N = 0
    torch.manual_seed(seed)
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs, _ = inputs.to(device), _.to(device)
        patches = F.unfold(inputs, patch_size).transpose(0, 1).contiguous()
        patches = patches.view(patches.size(0), -1).double() - mean
        n = inputs.size(0)
        batch_covariance = (patches @ patches.t() / patches.size(1))
        if covariance is None:
            covariance = batch_covariance
        else:
            covariance = N/(N+n)*covariance + n/(N+n)*batch_covariance
        N += n

    (eigvals, eigvecs) = scipy.linalg.eigh(covariance.cpu().numpy())
    inv_sqrt_eigvals = np.diag(np.power(eigvals + reg, -1/2))
    whitening_operator = eigvecs.dot(inv_sqrt_eigvals)

    return (mean.view(-1).cpu().numpy().astype('float32'),
            whitening_operator.astype('float32'))

def compute_whitening(patches, reg=0.001):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0

    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)

    patches_mean = np.mean(patches, axis=0, keepdims=True)

    patches = patches - patches_mean

    if reg < 0 :
        return (patches.reshape(orig_shape).astype('float32'),
                np.eye(patches.shape[1]).astype('float32'),
                patches_mean.reshape(-1).astype('float32'))

    covariance_matrix = patches.T.dot(patches) / patches.shape[0]

    (eigvals, eigvecs) = scipy.linalg.eigh(covariance_matrix)

    inv_sqrt_eigvals = np.diag(np.power(eigvals + reg, -1/2))

    whitening_operator = eigvecs.dot(inv_sqrt_eigvals)

    return (patches_mean.reshape(-1).astype('float32'),
            whitening_operator.astype('float32'))


def heaviside(x, bias):
    if x.dtype == torch.float16:
        return (x > bias).half()
    return (x > bias).float()


def topk(x, k):
    x_abs = torch.abs(x)
    if x.dtype == torch.float16:
        return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).half() * x
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).float() * x


def topk_heaviside(x, k):
    x_abs = torch.abs(x)
    if x.dtype == torch.float16:
        return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).half() * x.sign().half()
    return (x_abs >= x_abs.topk(dim=1, k=k).values.min(dim=1, keepdim=True).values).float() * x.sign().float()


def compute_classifier_outputs(outputs1, outputs2, targets, args, batch_norm1, batch_norm2, classifier1, classifier2, classifier, train=True):
    if args.batch_norm:
        outputs1, outputs2 = batch_norm1(outputs1), batch_norm2(outputs2)


    if args.convolutional_classifier == 0:
        outputs1, outputs2 = outputs1.view(outputs1.size(0),-1), outputs2.view(outputs2.size(0),-1)

    outputs1, outputs2 = classifier1(outputs1), classifier2(outputs2)
    outputs = outputs1 + outputs2


    if args.convolutional_classifier > 0:
        if args.bottleneck_dim > 0:
            outputs = classifier(outputs)
            outputs = F.adaptive_avg_pool2d(outputs, 1)
        else:
            outputs = F.adaptive_avg_pool2d(outputs, 1)
    elif args.bottleneck_dim > 0:
        outputs = classifier(outputs)

    outputs = outputs.view(outputs.size(0),-1)

    return outputs, targets


def create_classifier_blocks(out1, out2, args, params, n_classes):
    batch_norm1, batch_norm2, classifier1, classifier2, classifier =  None, None, None, None, None

    if args.batch_norm:
        batch_norm1 = nn.BatchNorm2d(out1.size(1), affine=(not args.no_affine_batch_norm)).to(device).float()
        batch_norm2 = nn.BatchNorm2d(out2.size(1), affine=(not args.no_affine_batch_norm)).to(device).float()
        params += list(batch_norm1.parameters()) + list(batch_norm2.parameters())


    if args.convolutional_classifier > 0:
        if args.bottleneck_dim > 0:
            classifier = nn.Conv2d(args.bottleneck_dim, n_classes, args.convolutional_classifier).to(device).float()
            params += list(classifier.parameters())
            classifier1 = nn.Conv2d(out1.size(1), args.bottleneck_dim, 1).to(device).float()
            classifier2 = nn.Conv2d(out2.size(1), args.bottleneck_dim, 1).to(device).float()
        else:
            classifier1 = nn.Conv2d(out1.size(1), n_classes, args.convolutional_classifier).to(device).float()
            classifier2 = nn.Conv2d(out2.size(1), n_classes, args.convolutional_classifier).to(device).float()
    else:
        out1, out2 = out1.view(out1.size(0), -1), out2.view(out1.size(0), -1)
        if args.bottleneck_dim > 0:
            classifier = nn.Linear(args.bottleneck_dim, n_classes).to(device).float()
            params += list(classifier.parameters())
            classifier1 = nn.Linear(out1.size(1), args.bottleneck_dim).to(device).float()
            classifier2 = nn.Linear(out2.size(1), args.bottleneck_dim).to(device).float()
        else:
            classifier1 = nn.Linear(out1.size(1), n_classes).to(device).float()
            classifier2 = nn.Linear(out2.size(1), n_classes).to(device).float()

    params += list(classifier1.parameters()) + list(classifier2.parameters())

    return batch_norm1, batch_norm2, classifier1, classifier2, classifier


def compute_channel_mean_and_std(loader, net, n_channel_convolution,
        kernel_convolution, whitening_operator,
        minus_whitened_patches_mean, n_epochs=1,
        seed=0):

    mean1, mean2 = torch.DoubleTensor(n_channel_convolution).fill_(0).to(device), torch.DoubleTensor(n_channel_convolution).fill_(0).to(device)
    std1, std2 = torch.DoubleTensor(n_channel_convolution).fill_(0).to(device), torch.DoubleTensor(n_channel_convolution).fill_(0).to(device)

    print(' first pass to compute the mean')
    N = 0
    torch.manual_seed(seed)
    for i_epoch in range(n_epochs):
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs, _ = inputs.to(device), _.to(device)

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
                n = inputs.size(0)
                mean1 = N/(N+n) * mean1 + outputs1.mean(dim=(0, 2, 3)).double() * n/(N+n)
                mean2 = N/(N+n) * mean2 + outputs2.mean(dim=(0, 2, 3)).double() * n/(N+n)
                N += n

    mean1 = mean1.view(1, -1, 1, 1)
    mean2 = mean2.view(1, -1, 1, 1)
    print(' second pass to compute the std')
    N = 0
    torch.manual_seed(seed)
    for i_epoch in range(n_epochs):
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs, _ = inputs.to(device), _.to(device)

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
                n = inputs.size(0)
                std1 = N/(N+n) * std1 + ((outputs1 - mean1)**2).mean(dim=(0, 2, 3)).double() * n/(N+n)
                std2 = N/(N+n) * std2 + ((outputs2 - mean2)**2).mean(dim=(0, 2, 3)).double() * n/(N+n)
                N += n
    std1, std2 = torch.sqrt(std1), torch.sqrt(std2)

    return mean1.float(), mean2.float(), std1.float().view(1, -1, 1, 1), std2.float().view(1, -1, 1, 1)
