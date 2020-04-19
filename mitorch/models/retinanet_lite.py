import collections
import math
import torch
from .retinanet import RetinaNet
from .modules import DepthwiseSeparableConv2d


class RetinaNetLite(RetinaNet):
    class DetectionBlock(RetinaNet.DetectionBlock):
        def _create_branch(self, in_channels, out_channels, num_blocks):
            return torch.nn.Sequential(collections.OrderedDict(
                [(f'conv{i}', DepthwiseSeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1)) for i in range(num_blocks)]
                + [(f'conv{num_blocks}', DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1, use_bn2=False, activation2='none'))]
            ))

        def reset_parameters(self):
            pi = 0.01
            self.conv_cls[-1].pointwise_conv.conv.bias.data.fill_(-math.log((1 - pi) / pi))

    def __init__(self, backbone, num_classes, prior_box=None, num_blocks=4):
        super().__init__(backbone, num_classes, prior_box, num_blocks, RetinaNetLite.DetectionBlock)
