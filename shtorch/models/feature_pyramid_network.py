import torch
from .model import Model


class FeaturePyramidNetwork(Model):
    BASE_FEATURE_NAMES = {'EfficientNetB0': ['features.conv1', 'features.block5_0.conv0', 'features.block4_0.conv'],
                          'MobileNetV2': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'], # stride 32, 16, 8
                          'MobileNetV3': ['features.conv1', 'features.block4_0.conv0', 'features.block3_0.conv0'],
                          'MobileNetV3Small': ['features.conv2', 'features.block3_0.conv0', 'features.block2_0.conv0'],
                          'SEResNext50': ['features.block3_2', 'features.block3_0.conv0', 'features.block2_0.conv0'],
                          'ShuffleNetV2': ['features.conv1', 'features.block2_0.conv0', 'features.block1_0.conv0'],
                          'VGG16ForSSD': ['features.conv5_1', 'features.conv3_2', 'features.conv2_2']
    }

    def __init__(self, backbone, out_channels=256):
        self.base_feature_names = self.BASE_FEATURE_NAMES.get(type(backbone).__name__, None)
        if not self.base_feature_names:
            raise NotImplementedError(f"FeaturePyramidNetwork: The backbone {type(backbone).__name__} is not supported")
        base_output_shapes = backbone.get_output_shapes(self.base_feature_names)
        super(FeaturePyramidNetwork, self).__init__([out_channels] * 5)

        self.backbone = backbone

        self.conv0_0 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1)
        self.conv0_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # P_5
        self.upsample0 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1_0 = torch.nn.Conv2d(base_output_shapes[1], out_channels, kernel_size=1)
        self.conv1_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # P_4
        self.upsample1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2_0 = torch.nn.Conv2d(base_output_shapes[2], out_channels, kernel_size=1)
        self.conv2_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # P_3
        self.upsample2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=3, padding=1, stride=2) # P_6

        self.conv4 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) # P_7

        self.activation = torch.nn.ReLU()

    def forward(self, input):
        base_features = self.backbone(input, self.base_feature_names)

        x = self.conv0_0(base_features[0])
        x = self.upsample0(x)
        p5 = self.conv0_1(x)

        x = self.conv1_0(base_features[1])
        x = self.upsample1(x)
        p4 = self.conv1_1(x)

        x = self.conv2_0(base_features[2])
        x = self.upsample2(x)
        p3 = self.conv2_1(x)

        p6 = self.conv3(base_features[0])
        x = self.activation(p6)
        p7 = self.conv4(x)

        return [p3, p4, p5, p6, p7]
