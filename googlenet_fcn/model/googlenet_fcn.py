import types

import numpy as np

import torch
import torch.hub as hub
import torch.nn as nn
from torchvision import models
from torchvision.models.googlenet import BasicConv2d, Inception

pretrained_models = {
    'cityscapes': {
        'url': 'https://github.com/TheCodez/pytorch-GoogLeNet-FCN/releases/download/1.0/googlenet_fcn_cs-c59aa8ae.pth',
        'num_classes': 19
    },
    'voc_aug': {
        'url': 'https://github.com/TheCodez/pytorch-GoogLeNet-FCN/releases/download/1.0/googlenet_fcn_voc-ab3f3182.pth',
        'num_classes': 21
    }
}


def googlenet_fcn(pretrained=None, num_classes=19):
    """Constructs a FCN using GoogLeNet.

    Args:
        pretrained (str, optional): If not ``None``, returns a pre-trained model. Possible values: ``cityscapes``
            and ``voc`` .
        num_classes (int): number of output classes
    """
    if pretrained is not None:
        model = GoogLeNetFCN(pretrained_models[pretrained]['num_classes'])
        model.load_state_dict(hub.load_state_dict_from_url(pretrained_models[pretrained]['url']))
        return model

    model = GoogLeNetFCN(num_classes)
    return model


class GoogLeNetFCN(nn.Module):

    def __init__(self, num_classes=19, transform_input=True):
        super(GoogLeNetFCN, self).__init__()
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.score3b = nn.Conv2d(480, num_classes, kernel_size=1)
        self.score4e = nn.Conv2d(832, num_classes, kernel_size=1)
        self.score5b = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                upsampling_weight = get_upsampling_weight(m.out_channels, m.kernel_size[0])
                with torch.no_grad():
                    m.weight.copy_(upsampling_weight)

    def init_from_googlenet(self):
        googlenet = models.googlenet(pretrained=True)
        self.load_state_dict(googlenet.state_dict(), strict=False)
        self.transform_input = True

    def forward(self, x):
        size = x.size()

        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        inception3b = self.inception3b(x)
        x = self.maxpool3(inception3b)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        inception4e = self.inception4e(x)

        x = self.maxpool4(inception4e)
        x = self.inception5a(x)
        inception5b = self.inception5b(x)

        score5b = self.score5b(inception5b)
        score4e = self.score4e(inception4e)
        semantics = self.upscore2(score5b)
        semantics = semantics[:, :, 1:1 + score4e.size(2), 1:1 + score4e.size(3)]
        semantics += score4e
        score3b = self.score3b(inception3b)
        semantics = self.upscore4(semantics)
        semantics = semantics[:, :, 1:1 + score3b.size(2), 1:1 + score3b.size(3)]
        semantics += score3b
        semantics = self.upscore8(semantics)
        semantics = semantics[:, :, 4:4 + size[2], 4:4 + size[3]].contiguous()

        return semantics


def get_upsampling_weight(channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling

    Based on: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    weight = torch.zeros([channels, channels, kernel_size, kernel_size], dtype=torch.float64)
    weight[range(channels), range(channels), :, :] = filt

    return weight


if __name__ == '__main__':
    num_classes, width, height = 20, 2048, 1024

    model = GoogLeNetFCN(num_classes)  # .to('cuda')
    inp = torch.randn(1, 3, height, width)  # .to('cuda')

    with torch.no_grad():
        sem = model(inp)
        print(sem.size())
        assert sem.size() == torch.Size([1, num_classes, height, width])

    print('Pass size check.')
