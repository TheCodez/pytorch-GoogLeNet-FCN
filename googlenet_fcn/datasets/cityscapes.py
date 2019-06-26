import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms.functional as F


class CityscapesDataset(datasets.Cityscapes):

    def __init__(self, root, split='train', mode='fine', test=False, transforms=None):
        super(CityscapesDataset, self).__init__(root, split, mode, target_type=['semantic', 'color'])

        self.test = test
        self.transforms = transforms

    def __getitem__(self, index):
        image, (target, color) = super(CityscapesDataset, self).__getitem__(index)

        if self.transforms:
            image, target = self.transforms(image, target)

        if self.test:
            color = F.to_tensor(color)
            return image, target, color
        else:
            return image, target

    @staticmethod
    def convert_id_to_train_id(target):
        target_copy = target.clone()

        for cls in CityscapesDataset.classes:
            target_copy[target == cls.id] = cls.train_id

        return target_copy

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


class FineCoarseDataset(data.Dataset):

    def __init__(self, fine, coarse):
        self.coarse = coarse
        self.fine = fine

        self.use_fine = True

    def __getitem__(self, index):
        if self.use_fine:
            idx = index % len(self.fine)
            ret = self.fine[idx]
        else:
            ret = self.coarse[index]

        self.use_fine = not self.use_fine
        return ret

    def __len__(self):
        return len(self.coarse)
