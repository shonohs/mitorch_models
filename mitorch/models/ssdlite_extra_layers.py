import torch
from .model import Model
from .modules.convolution import Conv2dAct


class SSDLiteExtraLayers(Model):
    BASE_FEATURE_NAMES = {'EfficientNetB0': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB1': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB2': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB3': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB4': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB5': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB6': ['features.block5_0.conv0', 'features.conv1'],
                          'EfficientNetB7': ['features.block5_0.conv0', 'features.conv1'],
                          'MobileNetV2': ['features.block5_0.conv0', 'features.conv1'],
                          'MobileNetV3': ['features.block4_0.conv0', 'features.conv2'],
                          'MobileNetV3Small': ['features.block3_0.conv0', 'features.conv2'],
                          'ResNext14': ['features.block3_0.conv0', 'features.block3_0'],
                          'ResNext26': ['features.block3_0.conv0', 'features.block3_1'],
                          'ResNext50': ['features.block3_0.conv0', 'features.block3_2'],
                          'ResNext101': ['features.block3_0.conv0', 'features.block3_2'],
                          'SEResNext14': ['features.block3_0.conv0', 'features.block3_0'],
                          'SEResNext26': ['features.block3_0.conv0', 'features.block3_1'],
                          'SEResNext50': ['features.block3_0.conv0', 'features.block3_2'],
                          'SEResNext101': ['features.block3_0.conv0', 'features.block3_2'],
                          'ShuffleNetV2': ['features.block2_0.conv0', 'features.conv1'],
                          'VGG16ForSSD': ['features.conv3_2', 'features.conv5_1']
    }

    def __init__(self, backbone, width_multiplier=1):
        m = width_multiplier
        self.base_feature_names = self.BASE_FEATURE_NAMES.get(type(backbone).__name__, None)
        if not self.base_feature_names:
            raise NotImplementedError(f"SSDLiteExtraLayers: The backbone {type(backbone).__name__} is not supported")
        base_output_shapes = backbone.get_output_shapes(self.base_feature_names)
        super(SSDLiteExtraLayers, self).__init__(base_output_shapes + [int(512 * m), int(256 * m), int(256 * m), int(128 * m)])

        self.base_model = backbone

        in_channels = backbone.output_dim

        self.conv0_0 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1, stride=2, groups=in_channels, use_bn=True)
        self.conv0_1 = Conv2dAct(in_channels, int(512 * m), kernel_size=1, use_bn=True)
        self.conv1_0 = Conv2dAct(int(512 * m), int(512 * m), kernel_size=3, padding=1, stride=2, groups=int(512 * m), use_bn=True)
        self.conv1_1 = Conv2dAct(int(512 * m), int(256 * m), kernel_size=1, use_bn=True)
        self.conv2_0 = Conv2dAct(int(256 * m), int(256 * m), kernel_size=3, padding=1, stride=2, groups=int(256 * m), use_bn=True)
        self.conv2_1 = Conv2dAct(int(256 * m), int(256 * m), kernel_size=1, use_bn=True)
        self.conv3_0 = Conv2dAct(int(256 * m), int(256 * m), kernel_size=3, padding=1, stride=2, groups=int(256 * m), use_bn=True)
        self.conv3_1 = Conv2dAct(int(256 * m), int(128 * m), kernel_size=1, use_bn=True)

    def forward(self, input):
        features = self.base_model.forward(input, self.base_feature_names)
        f0 = self.conv0_1(self.conv0_0(features[-1]))
        f1 = self.conv1_1(self.conv1_0(f0))
        f2 = self.conv2_1(self.conv2_0(f1))
        f3 = self.conv3_1(self.conv3_0(f2))
        return features + [f0, f1, f2, f3]
