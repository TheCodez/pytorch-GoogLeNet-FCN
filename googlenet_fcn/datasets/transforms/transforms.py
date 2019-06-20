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


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        min_h, min_w = self.min_size
        max_h, max_w = self.max_size

        h = random.randint(int(min_h), int(max_h))
        w = random.randint(int(min_w), int(max_w))

        image = F.resize(image, (h, w))
        target = F.resize(target, (h, w), interpolation=Image.NEAREST)
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

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
    def __init__(self, p=0.5, degress=(0, 0), translate=None, scale=None, shear=None):
        self.p = p
        self.degrees = degress
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img, target):
        if random.random() < self.p:
            angle, translations, scale, shear = T.RandomAffine.get_params(self.degrees, self.translate, self.scale,
                                                                          self.shear, img.size)
            img = F.affine(img, angle, translations, scale, shear, resample=False)
            target = F.affine(target, angle, translations, scale, shear, resample=False, fillcolor=255)

        return img, target


class RandomGaussionNoise(object):
    def __init__(self, p=0.5, mean=0.0, std=0.1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        if random.random() < self.p:
            noise = img.clone().normal_(self.mean, self.std)
            img = img + noise

        return img, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target
