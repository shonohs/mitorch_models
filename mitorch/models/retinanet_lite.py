import collections
import math
import torch
from .retinanet import RetinaNet
from .modules import Conv2dAct, ModuleBase


class RetinaNetLite(RetinaNet):
    class DetectionBlock(RetinaNet.DetectionBlock):
        def _create_branch(self, in_channels, out_channels, num_blocks):
            layers = collections.OrderedDict()
            for i in range(num_blocks):
                layers[f'conv{i}_0'] = Conv2dAct(in_channels, in_channels, kernel_size=1)
                layers[f'conv{i}_1'] = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            layers[f'conv{num_blocks}_0'] = Conv2dAct(in_channels, out_channels, kernel_size=1)
            layers[f'conv{num_blocks}_1'] = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)

            return torch.nn.Sequential(layers)

    def __init__(self, backbone, num_classes, prior_box = None, num_blocks = 4):
        super(RetinaNetLite, self).__init__(backbone, num_classes, prior_box, num_blocks, RetinaNetLite.DetectionBlock)
