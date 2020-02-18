import torch
from .model import Model
from .modules import Add, Conv2dAct


class FeaturePyramidNetwork(Model):
    BASE_FEATURE_NAMES = {'EfficientNetB0': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB1': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB2': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB3': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB4': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB5': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB6': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'EfficientNetB7': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'],
                          'MobileNetV2': ['features.conv1', 'features.block5_0.conv0', 'features.block3_0.conv0'], # stride 32, 16, 8
                          'MobileNetV3': ['features.conv1', 'features.block4_0.conv0', 'features.block3_0.conv0'],
                          'MobileNetV3Small': ['features.conv1', 'features.block3_0.conv0', 'features.block2_0.conv0'],
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

        self.base_model = backbone

        self.conv0_0 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1)
        self.conv0_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # P_5

        self.conv1_0 = torch.nn.Conv2d(base_output_shapes[1], out_channels, kernel_size=1)
        self.conv1_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # P_4

        self.conv2_0 = torch.nn.Conv2d(base_output_shapes[2], out_channels, kernel_size=1)
        self.conv2_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # P_3

        self.conv3 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=3, padding=1, stride=2) # P_6

        self.conv4 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) # P_7

        self.add0 = Add()
        self.add1 = Add()
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        base_features = self.base_model(input, self.base_feature_names)

        # Coarsest first. C5, C4, C3
        assert [b.shape[2] for b in base_features] == sorted([b.shape[2] for b in base_features])
        assert [b.shape[3] for b in base_features] == sorted([b.shape[3] for b in base_features])

        x = self.conv0_0(base_features[0]) # C5
        p5 = self.conv0_1(x)
        x = torch.nn.functional.interpolate(x, base_features[1].size()[2:]) # Upsample 2x

        x = self.add0(x, self.conv1_0(base_features[1])) # C4
        p4 = self.conv1_1(x)
        x = torch.nn.functional.interpolate(x, base_features[2].size()[2:]) # Upsample 2x

        x = self.add1(x, self.conv2_0(base_features[2])) # C3
        p3 = self.conv2_1(x)

        p6 = self.conv3(base_features[0]) # C5

        x = self.activation(p6)
        p7 = self.conv4(x)

        return [p3, p4, p5, p6, p7]


class FeaturePyramidNetworkLite(Model):
    def __init__(self, backbone, out_channels=256):
        self.base_feature_names = FeaturePyramidNetwork.BASE_FEATURE_NAMES.get(type(backbone).__name__, None)
        if not self.base_feature_names:
            raise NotImplementedError(f"FeaturePyramidNetwork: The backbone {type(backbone).__name__} is not supported")
        base_output_shapes = backbone.get_output_shapes(self.base_feature_names)
        super(FeaturePyramidNetworkLite, self).__init__([out_channels] * 5)

        self.base_model = backbone

        self.conv0_0 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1)
        self.conv0_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv0_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1) # P_5

        self.conv1_0 = torch.nn.Conv2d(base_output_shapes[1], out_channels, kernel_size=1)
        self.conv1_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv1_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1) # P_4

        self.conv2_0 = torch.nn.Conv2d(base_output_shapes[2], out_channels, kernel_size=1)
        self.conv2_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv2_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1) # P_3

        self.conv3_0 = Conv2dAct(base_output_shapes[0], base_output_shapes[0], kernel_size=3, padding=1, stride=2, groups=base_output_shapes[0])
        self.conv3_1 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1) # P_6

        self.conv4_0 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, stride=2, groups=out_channels)
        self.conv4_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1) # P_7

        self.add0 = Add()
        self.add1 = Add()
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        base_features = self.base_model(input, self.base_feature_names)

        # Coarsest first. C5, C4, C3
        assert [b.shape[2] for b in base_features] == sorted([b.shape[2] for b in base_features])
        assert [b.shape[3] for b in base_features] == sorted([b.shape[3] for b in base_features])

        x = self.conv0_0(base_features[0]) # C5
        p5 = self.conv0_2(self.conv0_1(x))
        x = torch.nn.functional.interpolate(x, base_features[1].size()[2:]) # Upsample 2x

        x = self.add0(x, self.conv1_0(base_features[1])) # C4
        p4 = self.conv1_2(self.conv1_1(x))
        x = torch.nn.functional.interpolate(x, base_features[2].size()[2:]) # Upsample 2x

        x = self.add1(x, self.conv2_0(base_features[2])) # C3
        p3 = self.conv2_2(self.conv2_1(x))

        x = self.conv3_0(base_features[0]) # C5
        p6 = self.conv3_1(x)

        x = self.activation(p6)
        p7 = self.conv4_1(self.conv4_0(x))

        return [p3, p4, p5, p6, p7]
