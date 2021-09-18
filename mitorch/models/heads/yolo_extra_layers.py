import torch
from .head import Head
from ..modules import Conv2dAct


class YoloV2ExtraLayers(Head):
    def __init__(self, backbone):
        super().__init__(backbone, [5, 4], [1024])
        base_output_shapes = backbone.get_output_shapes(self.base_feature_names)

        self.conv0 = Conv2dAct(base_output_shapes[0], 1024, kernel_size=3, padding=1, activation='leaky_relu')
        self.conv1 = Conv2dAct(1024, 1024, kernel_size=3, padding=1, activation='leaky_relu')
        self.conv1_0 = Conv2dAct(base_output_shapes[1], 64, kernel_size=1, activation='leaky_relu')
        self.conv2 = Conv2dAct(1024, 1024, kernel_size=3, padding=1, activation='leaky_relu')

    def forward(self, input_tensor):
        features = self.get_base_features(input_tensor)
        b = self.conv1(self.conv0(features[0]))  # 13x13
        b2 = self.conv1_0(features[1])  # 26x26
        n, c, h, w = b2.shape
        reorged = b2.view(n, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).view(n, c * 4, h, w)
        concatenated = torch.cat((b, reorged), 1)
        return self.conv2(concatenated)


class TinyYoloV2ExtraLayers(Head):
    def __init__(self, backbone):
        super().__init__(backbone, [5], [512])
        base_output_shapes = backbone.get_output_shapes(self.base_feature_names)

        self.conv0 = Conv2dAct(base_output_shapes[0], 512, kernel_size=3, padding=1, activation='leaky_relu')

    def forward(self, input_tensor):
        features = self.get_base_features(input_tensor)
        return self.conv0(features[0])
