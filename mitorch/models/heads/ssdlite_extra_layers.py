from .head import Head
from ..modules import DepthwiseSeparableConv2d


class SSDLiteExtraLayers(Head):
    def __init__(self, backbone, width_multiplier=1):
        m = width_multiplier
        base_output_shapes = Head.get_base_output_shapes(backbone, [4, 5])
        out_channels = base_output_shapes + [int(512 * m), int(256 * m), int(256 * m), int(128 * m)]
        super().__init__(backbone, [4, 5], out_channels)
        assert len(out_channels) == 6

        self.conv0 = DepthwiseSeparableConv2d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1)
        self.conv1 = DepthwiseSeparableConv2d(out_channels[2], out_channels[3], kernel_size=3, stride=2, padding=1)
        self.conv2 = DepthwiseSeparableConv2d(out_channels[3], out_channels[4], kernel_size=3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv2d(out_channels[4], out_channels[5], kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        features = self.get_base_features(input)
        f0 = self.conv0(features[-1])
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return features + [f0, f1, f2, f3]
