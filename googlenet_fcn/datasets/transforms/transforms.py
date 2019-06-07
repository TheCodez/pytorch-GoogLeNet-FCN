import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

from googlenet_fcn.datasets.cityscapes import CityscapesDataset


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor(object):

    def __call__(self, img, target):
        img = F.to_tensor(img)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)

        return img, target


class ConvertIdToTrainId(object):

    def __call__(self, img, target):
        target = CityscapesDataset.convert_id_to_train_id(target)

        return img, target


class RandomApply(object):

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, target):
        if self.p < random.random():
            return img, target

        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Resize(object):

    def __init__(self, new_size):
        self.old_size = (1024, 2048)
        self.new_size = new_size

        self.xscale = self.new_size[1] / self.old_size[1]
        self.yscale = self.new_size[0] / self.old_size[0]

    def __call__(self, img, target):
        img = F.resize(img, self.new_size, interpolation=Image.BILINEAR)
        target = F.resize(target, self.new_size, interpolation=Image.NEAREST)

        return img, target


class Rescale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, target):
        width, height = img.size
        width *= self.scale
        height *= self.scale

        new_size = (int(height), int(width))
        img = F.resize(img, new_size, interpolation=Image.BILINEAR)
        target = F.resize(target, new_size, interpolation=Image.NEAREST)

        return img, target


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = F.hflip(target)

        return img, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        from torchvision import transforms
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target):
        img = self.transform(img)

        return img, target


class RandomGaussionBlur(object):
    def __init__(self, p=0.5, radius=0.8):
        self.p = p
        self.radius = radius

    def __call__(self, img, target):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))

        return img, target


class RandomAffine(object):
    def __init__(self, scale=None, shear=None):
        self.scale = scale
        self.shear = shear

    def __call__(self, img, target):
        angle, translations, scale, shear = T.RandomAffine.get_params((0, 0), None, self.scale, self.shear, img.size)
        img = F.affine(img, angle, translations, scale, shear, resample=False)
        target = F.affine(target, angle, translations, scale, shear, resample=False, fillcolor=255)

        return img, target


class RandomGaussionNoise(object):
    def __init__(self, scale=0.02):
        self.scale = scale

    def __call__(self, img, inst):
        img = img + self.scale * torch.randn_like(img)

        return img, inst


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
