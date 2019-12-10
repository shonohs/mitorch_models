"""VGG (https://arxiv.org/abs/1409.1556)"""
import collections
import torch
from .model import Model
from .modules import Conv2dAct, LinearAct

class VGG(Model):
    INPUT_SIZE = 224

    def forward(self, input):
        return self.features(input)


class VGG_A(VGG):
    def __init__(self):
        super(VGG_A, self).__init__(4096, use_bn=False)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv3', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv5', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv6', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv7', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_B(VGG):
    def __init__(self):
        super(VGG_B, self).__init__(4096, use_bn=False)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv3', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv5', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv6', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv7', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv8', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv9', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_C(VGG):
    def __init__(self):
        super(VGG_C, self).__init__(4096, use_bn=False)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv3', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv5', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv6', Conv2dAct(256, 256, kernel_size=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv7', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv8', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv9', Conv2dAct(512, 512, kernel_size=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv10', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv11', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv12', Conv2dAct(512, 512, kernel_size=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_D(VGG):
    def __init__(self):
        super(VGG_D, self).__init__(4096, use_bn=False)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv3', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv5', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv6', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv7', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv8', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv9', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv10', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv11', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv12', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_E(VGG):
    def __init__(self):
        super(VGG_E, self).__init__(4096, use_bn=False)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv3', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv5', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv6', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv7', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv8', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv9', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv10', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv11', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv12', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv13', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv14', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv15', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG16(VGG_D):
    pass
