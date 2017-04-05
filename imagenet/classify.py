#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-04-04 21:46
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def make_dataset(dir, type):
    if type == 0:
        class_file = os.join(dir, 'train.txt')
    elif type == 1:
        class_file = os.join(dir, 'test.txt')
    else:
        class_file = os.join(dir, 'val.txt')

    with open(class_file, 'r') as f:
        ret = []
        for l in f:
            elem = l.strip().split(' ')
        ret.append((elem[0].strip(), elem[1].strip()))
    return ret



def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageNet(data.Dataset):
    def __init__(self, root, type=0, transform=None, target_transform=None,
                 loader=default_loader):
        '''
        `type`: 0 - train, 1 - test, 2 - validate
        '''
        imgs = make_dataset(root, type)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
