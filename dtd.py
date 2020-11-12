import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

SPLIT = '1'


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtnames, datadir, class_to_idx):
    images = []
    labels = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'images', line.strip())
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[classname])

    return images, labels


class DTDDataloader(data.Dataset):
    def __init__(self, path='DTD', transform=None, train=True):
        classes, class_to_idx = find_classes(os.path.join(path, 'images'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform

        if train:
            filename = [os.path.join(path, 'labels/train' + SPLIT + '.txt'),
                        os.path.join(path, 'labels/val' + SPLIT + '.txt')]
        else:
            filename = [os.path.join(path, 'labels/test' + SPLIT + '.txt')]

        self.images, self.labels = make_dataset(filename, path, class_to_idx)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)


class Dataloder():
    def __init__(self, path, spatial_size, batchsize):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train_norandom = transforms.Compose([
            transforms.Resize(9*spatial_size//8),
            transforms.CenterCrop(spatial_size),
            transforms.ToTensor(),
            normalize,
        ])
        transform_train = transforms.Compose([
            transforms.Resize(9*spatial_size//8),
            # transforms.RandomCrop(spatial_size),
            transforms.RandomResizedCrop(spatial_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(9*spatial_size//8),
            transforms.CenterCrop(spatial_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset_norandom = DTDDataloader(path, transform_train_norandom, train=True)
        trainset = DTDDataloader(path, transform_train, train=True)
        testset = DTDDataloader(path, transform_test, train=False)

        kwargs = {'num_workers': 8, 'pin_memory': True}
        trainloader_norandom = torch.utils.data.DataLoader(trainset_norandom, batch_size=batchsize, shuffle=False, **kwargs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
        batchsize, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
       batchsize, shuffle=False, **kwargs)
        self.classes = trainset.classes
        self.trainset = trainset
        self.testset = testset
        self.trainloader_norandom = trainloader_norandom
        self.trainloader = trainloader
        self.testloader = testloader

    def getloader(self):
        return self.classes, self.trainset, self.testset, self.trainloader, self.testloader, self.trainloader_norandom
