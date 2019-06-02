import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


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

    def __call__(self, img, inst):
        img = self.transform(img)

        return img, inst


"""
class RandomGaussionBlur(object):
    def __init__(self, sigma=(0.15, 1.15)):
        self.sigma = sigma

    def __call__(self, img, inst):
        sigma = self.sigma[0] + random.random() * self.sigma[1]
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        img = Image.fromarray(blurred_img.astype(np.uint8))

        return img, inst
"""


class RandomAffine(object):
    def __init__(self, scale=None, shear=None, fillcolor=0):
        self.scale = scale
        self.shear = shear
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(scale_ranges, shears):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return scale, shear

    def __call__(self, img, inst):
        scale, shear = self.get_params(self.scale, self.shear)
        img = F.affine(img, 0, (0, 0), scale, shear, resample=False, fillcolor=self.fillcolor)
        inst = F.affine(inst, 0, (0, 0), scale, shear, resample=False, fillcolor=self.fillcolor)

        return img, inst


class RandomGaussionNoise(object):
    def __init__(self, scale=0.02):
        self.scale = scale

    def __call__(self, img, inst):
        img = img + self.scale * torch.randn_like(img)

        return img, inst


class Normalize(object):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, img, inst):
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, inst
