"""MnasFPN: Learning Latency-aware Pyramid Architecture for Object Detection on Mobile Devices
   (https://arxiv.org/pdf/1912.01106.pdf)

This is an implementation of the figure 2 architecture in the paper.
"""
import collections
import torch
from ..model import Model
from ..modules import Conv2dAct, DepthwiseSeparableConv2d
from .head import Head


class MnasFPN(Head):
    class BasicBlock(Model):
        def __init__(self, in_channels, feature_channels, kernel_size, in_scales, out_scale):
            super().__init__(in_channels)
            assert len(in_scales) == 2
            assert all(s in [3, 4, 5, 6] for s in in_scales)
            assert out_scale in [3, 4, 5, 6]

            self.in_scales = in_scales
            self.out_scale = out_scale
            self.conv_in0 = torch.nn.Conv2d(in_channels, feature_channels, kernel_size=1)
            self.conv_in1 = torch.nn.Conv2d(in_channels, feature_channels, kernel_size=1)
            self.conv0 = torch.nn.Conv2d(feature_channels, feature_channels, kernel_size, padding=kernel_size // 2, groups=feature_channels)
            self.conv1 = torch.nn.Conv2d(feature_channels, in_channels, kernel_size=1)
            self.relu = torch.nn.ReLU(inplace=True)

        def forward(self, in0, in1, out_shape):
            in0 = self._size_dependent_ordering(in0, self.conv_in0, self.in_scales[0], self.out_scale, out_shape)
            in1 = self._size_dependent_ordering(in1, self.conv_in1, self.in_scales[1], self.out_scale, out_shape)
            return self.conv1(self.conv0(self.relu(in0 + in1)))

        @staticmethod
        def _size_dependent_ordering(input, conv, in_scale, out_scale, out_shape):
            if in_scale < out_scale:  # Needs downsampling
                return conv(torch.nn.functional.interpolate(input, size=out_shape))
            elif in_scale > out_scale:  # Needs upsampling
                return torch.nn.functional.interpolate(conv(input), size=out_shape)
            else:  # in_scale == out_scale
                return conv(input)

    class BasicCell(Model):
        def __init__(self):
            super().__init__([48, 48, 48, 48])
            self.conv0 = MnasFPN.BasicBlock(48, 256, 3, [4, 5], 4)
            self.conv1 = MnasFPN.BasicBlock(48, 128, 3, [3, 4], 3)
            self.conv2 = MnasFPN.BasicBlock(48, 128, 3, [3, 4], 4)
            self.conv3 = MnasFPN.BasicBlock(48, 128, 5, [4, 6], 5)
            self.conv4 = MnasFPN.BasicBlock(48, 96, 3, [4, 6], 6)

        def forward(self, inputs):
            input3, input4, input5, input6 = inputs

            out0 = self.conv0(input4, input5, input4.shape[2:])
            out1 = self.conv1(input3, out0, input3.shape[2:])
            out2 = self.conv2(out1, out0, input4.shape[2:])
            out3 = self.conv3(out0, input6, input5.shape[2:])
            out4 = self.conv4(out0, input6, input6.shape[2:])

            return [input3 + out1, input4 + out2, input5 + out3, input6 + out4]

    def __init__(self, base_model, num_blocks=5):
        super(MnasFPN, self).__init__(base_model, [3, 4, 5], [48, 48, 48, 48])
        base_output_shapes = Head.get_base_output_shapes(base_model, [3, 4, 5])
        self.basic_blocks = torch.nn.Sequential(collections.OrderedDict([
            (f'block{i}', MnasFPN.BasicCell()) for i in range(num_blocks)]))

        self.conv0 = Conv2dAct(base_output_shapes[0], 48, kernel_size=1)
        self.conv1 = Conv2dAct(base_output_shapes[1], 48, kernel_size=1)
        self.conv2 = Conv2dAct(base_output_shapes[2], 48, kernel_size=1)
        self.conv3 = DepthwiseSeparableConv2d(base_output_shapes[2], 48, kernel_size=3, padding=1, stride=2)

    def forward(self, input):
        base_features = self.get_base_features(input)
        assert len(base_features) == 3

        features = [self.conv0(base_features[0]), self.conv1(base_features[1]), self.conv2(base_features[2]), self.conv3(base_features[2])]
        return self.basic_blocks(features)
