import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, (3, 3), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, (3, 3), (1, 1), (1, 1))
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor):
        identity = x

        if len(x.shape) != 4:
            raise RuntimeError("输入ResNetBlock输入张量纬度不合法！")
        else:
            out = F.relu(self.conv1(x))
            out = self.bn1(out)
            out = F.relu(self.conv2(out))
            out = self.bn2(out)
            out += identity
            return F.relu(out)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.res1 = ResNetBlock(1, 32)
        self.pool1 = nn.MaxPool2d((2,2),(2,2))
        self.res2 = ResNetBlock(32, 32)
        self.pool2 = nn.MaxPool2d((2,2),(2,2))
        self.res3 = ResNetBlock(32, 64)
        self.flatten = nn.Flatten()

    def forward(self, x:torch.Tensor):
        '''
        :param x: shape(1, 1,48,48)
        :return:
        '''
        out = self.res1(x)
        out = self.pool1(out)
        out = self.res2(out)
        out = self.pool2(out)
        out = self.res3(out)
        out = self.flatten(out)

