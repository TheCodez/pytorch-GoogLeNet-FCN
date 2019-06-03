import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image

from googlenet_fcn.datasets.transforms.transforms import Resize, RandomHorizontalFlip, Compose, ToTensor


class CityscapesDataset(datasets.Cityscapes):

    def __init__(self, root, split='train', mode='fine', joint_transform=None, img_transform=None):
        super(CityscapesDataset, self).__init__(root, split, mode, target_type='semantic')

        self.joint_transform = joint_transform
        self.img_transform = img_transform

    def __getitem__(self, index):
        image, target = super(CityscapesDataset, self).__getitem__(index)
        target = self.convert_id_to_train_id(target)

        if self.joint_transform:
            image, target = self.joint_transform(image, target)

        if self.img_transform:
            image = self.img_transform(image)

        return image, target

    @staticmethod
    def convert_id_to_train_id(target):
        target = np.array(target)
        target_copy = target.copy()

        for cls in CityscapesDataset.classes:
            target_copy[target == cls.id] = cls.train_id
            target = Image.fromarray(target_copy.astype(np.uint8))

        return target

    @staticmethod
    def convert_train_id_to_id(target):
        target_copy = target.clone()

        for cls in CityscapesDataset.classes:
            target_copy[target == cls.train_id] = cls.id

        return target_copy

    @staticmethod
    def get_class_from_name(name):
        for cls in CityscapesDataset.classes:
            if cls.name == name:
                return cls
        return None

    @staticmethod
    def get_class_from_id(id):
        for cls in CityscapesDataset.classes:
            if cls.id == id:
                return cls
        return None

    @staticmethod
    def get_colormap():
        cmap = torch.zeros([256, 3], dtype=torch.uint8)

        for cls in CityscapesDataset.classes:
            cmap[cls.id, :] = torch.tensor(cls.color)

        return cmap

    @staticmethod
    def num_classes():
        return len([cls for cls in CityscapesDataset.classes if not cls.ignore_in_eval])


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def visualize_segmentation(segmentation):
        cmap = CityscapesDataset.get_colormap()

        image = segmentation.squeeze(0)
        size = image.size()
        if len(size) > 2:
            color_image = torch.zeros([3, size[1], size[2]], dtype=torch.uint8)
        else:
            color_image = torch.zeros([3, size[0], size[1]], dtype=torch.uint8)

        for label in range(1, len(cmap)):
            mask = image == label

            color_image[0][mask] = cmap[label][0]
            color_image[1][mask] = cmap[label][1]
            color_image[2][mask] = cmap[label][2]

        return color_image


    joint_transforms = Compose([
        Resize((512, 1024)),
        RandomHorizontalFlip(),
        ToTensor(),
        # Normalize()
    ])

    print('Num classes: ', CityscapesDataset.num_classes())

    dataset = CityscapesDataset('../data/cityscapes', split='train', joint_transform=joint_transforms)
    # 2700 is Ulm University
    img, inst = dataset[2700]

    print('Instance size: ', inst.size())
    print('Num classes: ', dataset.num_classes())

    inst = CityscapesDataset.convert_train_id_to_id(inst)
    inst = visualize_segmentation(inst)

    inst = torchvision.transforms.functional.to_pil_image(inst)
    plt.imshow(inst)
    ax = plt.gca()
    plt.show()
