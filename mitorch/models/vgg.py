"""VGG (https://arxiv.org/abs/1409.1556)"""
import collections
import torch
from .model import Model
from .modules import Conv2dAct, LinearAct, default_module_settings


class VGG(Model):
    pass


class VGG_A(VGG):
    @default_module_settings(use_bn=False)
    def __init__(self):
        super().__init__(4096)
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
    @default_module_settings(use_bn=False)
    def __init__(self):
        super().__init__(4096)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0_0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv0_1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1_0', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv1_2', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_0', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv2_1', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_0', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv3_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4_0', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_C(VGG):
    @default_module_settings(use_bn=False)
    def __init__(self):
        super().__init__(4096)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0_0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv0_1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1_0', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv1_1', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_0', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv2_1', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv2_2', Conv2dAct(256, 256, kernel_size=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_0', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv3_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv3_2', Conv2dAct(512, 512, kernel_size=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4_0', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_2', Conv2dAct(512, 512, kernel_size=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_D(VGG):
    @default_module_settings(use_bn=False)
    def __init__(self):
        super().__init__(4096)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0_0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv0_1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1_0', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv1_1', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_0', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv2_1', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv2_2', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_0', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv3_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv3_2', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4_0', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_2', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG_E(VGG):
    @default_module_settings(use_bn=False)
    def __init__(self):
        super().__init__(4096)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0_0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv0_1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1_0', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv1_2', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_0', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv2_1', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv2_2', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv2_3', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_0', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv3_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv3_2', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv3_3', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4_0', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_2', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_3', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', torch.nn.Flatten()),
            ('fc0', LinearAct(7 * 7 * 512, 4096)),
            ('fc1', LinearAct(4096, 4096))]))


class VGG16(VGG_D):
    pass


class VGG16ForSSD(VGG):
    """SSD modification of VGG16. Changed pool4 and replaced fc0 and fc1"""
    @default_module_settings(use_bn=False)
    def __init__(self):
        super().__init__(1024)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0_0', Conv2dAct(3, 64, kernel_size=3, padding=1)),
            ('conv0_1', Conv2dAct(64, 64, kernel_size=3, padding=1)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1_0', Conv2dAct(64, 128, kernel_size=3, padding=1)),
            ('conv1_1', Conv2dAct(128, 128, kernel_size=3, padding=1)),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2_0', Conv2dAct(128, 256, kernel_size=3, padding=1)),
            ('conv2_1', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('conv2_2', Conv2dAct(256, 256, kernel_size=3, padding=1)),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3_0', Conv2dAct(256, 512, kernel_size=3, padding=1)),
            ('conv3_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv3_2', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4_0', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_1', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('conv4_2', Conv2dAct(512, 512, kernel_size=3, padding=1)),
            ('pool4', torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)),
            ('conv5_0', Conv2dAct(512, 1024, kernel_size=3, padding=1)),
            ('conv5_1', Conv2dAct(1024, 1024, kernel_size=1))]))
