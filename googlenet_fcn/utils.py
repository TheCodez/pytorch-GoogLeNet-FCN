import os
import tempfile
import types

import torch
import torch.nn as nn


def visualize_seg(image, cmap):
    out = torch.zeros([3, image.size(0), image.size(1)], dtype=torch.uint8)

    for label in range(1, len(cmap)):
        mask = image == label

        out[0, mask] = cmap[label, 0]
        out[1, mask] = cmap[label, 1]
        out[2, mask] = cmap[label, 2]

    return out


def save(obj, dir, file_name):
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.expanduser(dir))

    try:
        torch.save(obj, tmp.file)
    except BaseException:
        tmp.close()
        os.remove(tmp.name)
        raise
    else:
        tmp.close()
        os.rename(tmp.name, os.path.join(dir, file_name))


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def freeze_batchnorm(module):
    def train(self, mode=True):
        self.training = mode
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            else:
                m.train(mode)
        return self

    module.train = types.MethodType(train, module)

    return module
