"""
YOLO9000: Better, Faster, Stronger (https://arxiv.org/pdf/1612.08242)
"""
import collections
import torch
from .model import Model
from .modules import Conv2dAct


class Darknet19(Model):
    def __init__(self):
        super().__init__(1024)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 32, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', Conv2dAct(32, 64, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(64, 128, kernel_size=3, padding=1, activation='leaky_relu')),
            ('conv3', Conv2dAct(128, 64, kernel_size=1, activation='leaky_relu')),
            ('conv4', Conv2dAct(64, 128, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv5', Conv2dAct(128, 256, kernel_size=3, padding=1, activation='leaky_relu')),
            ('conv6', Conv2dAct(256, 128, kernel_size=1, activation='leaky_relu')),
            ('conv7', Conv2dAct(128, 256, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv8', Conv2dAct(256, 512, kernel_size=3, padding=1, activation='leaky_relu')),
            ('conv9', Conv2dAct(512, 256, kernel_size=1, activation='leaky_relu')),
            ('conv10', Conv2dAct(256, 512, kernel_size=3, padding=1, activation='leaky_relu')),
            ('conv11', Conv2dAct(512, 256, kernel_size=1, activation='leaky_relu')),
            ('conv12', Conv2dAct(256, 512, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv13', Conv2dAct(512, 1024, kernel_size=3, padding=1, activation='leaky_relu')),
            ('conv14', Conv2dAct(1024, 512, kernel_size=1, activation='leaky_relu')),
            ('conv15', Conv2dAct(512, 1024, kernel_size=3, padding=1, activation='leaky_relu')),
            ('conv16', Conv2dAct(1024, 512, kernel_size=1, activation='leaky_relu')),
            ('conv17', Conv2dAct(512, 1024, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool5', torch.nn.AdaptiveAvgPool2d(1)),
            ('flatten', torch.nn.Flatten())]))


class TinyDarknet(Model):
    """Base model for TinyYoloV2"""
    def __init__(self):
        super().__init__(1024)
        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, 16, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool0', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', Conv2dAct(16, 32, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', Conv2dAct(32, 64, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3', Conv2dAct(64, 128, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4', Conv2dAct(128, 256, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv5', Conv2dAct(256, 512, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool5', torch.nn.MaxPool2d(kernel_size=2, stride=1)),
            ('conv6', Conv2dAct(512, 1024, kernel_size=3, padding=1, activation='leaky_relu')),
            ('pool6', torch.nn.AdaptiveAvgPool2d(1)),
            ('flatten', torch.nn.Flatten())]))
