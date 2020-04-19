import torch
from .head import Head
from ..modules import Add, Conv2dAct


class FeaturePyramidNetwork(Head):
    def __init__(self, backbone, out_channels=256):
        super().__init__(backbone, [5, 4, 3], [out_channels] * 5)
        base_output_shapes = backbone.get_output_shapes(self.base_feature_names)

        self.conv0_0 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1)
        self.conv0_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # P_5

        self.conv1_0 = torch.nn.Conv2d(base_output_shapes[1], out_channels, kernel_size=1)
        self.conv1_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # P_4

        self.conv2_0 = torch.nn.Conv2d(base_output_shapes[2], out_channels, kernel_size=1)
        self.conv2_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # P_3

        self.conv3 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=3, padding=1, stride=2)  # P_6

        self.conv4 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)  # P_7

        self.add0 = Add()
        self.add1 = Add()
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        base_features = self.get_base_features(input)

        # Coarsest first. C5, C4, C3
        assert [b.shape[2] for b in base_features] == sorted([b.shape[2] for b in base_features])
        assert [b.shape[3] for b in base_features] == sorted([b.shape[3] for b in base_features])

        x = self.conv0_0(base_features[0])  # C5
        p5 = self.conv0_1(x)
        x = torch.nn.functional.interpolate(x, base_features[1].size()[2:])  # Upsample 2x

        x = self.add0(x, self.conv1_0(base_features[1]))  # C4
        p4 = self.conv1_1(x)
        x = torch.nn.functional.interpolate(x, base_features[2].size()[2:])  # Upsample 2x

        x = self.add1(x, self.conv2_0(base_features[2]))  # C3
        p3 = self.conv2_1(x)

        p6 = self.conv3(base_features[0])  # C5

        x = self.activation(p6)
        p7 = self.conv4(x)

        return [p3, p4, p5, p6, p7]


class FeaturePyramidNetworkLite(Head):
    def __init__(self, backbone, out_channels=256):
        super().__init__(backbone, [5, 4, 3], [out_channels] * 5)
        base_output_shapes = Head.get_base_output_shapes(backbone, [5, 4, 3])

        self.conv0_0 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1)
        self.conv0_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv0_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1)  # P_5

        self.conv1_0 = torch.nn.Conv2d(base_output_shapes[1], out_channels, kernel_size=1)
        self.conv1_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv1_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1)  # P_4

        self.conv2_0 = torch.nn.Conv2d(base_output_shapes[2], out_channels, kernel_size=1)
        self.conv2_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv2_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1)  # P_3

        self.conv3_0 = Conv2dAct(base_output_shapes[0], base_output_shapes[0], kernel_size=3, padding=1, stride=2, groups=base_output_shapes[0])
        self.conv3_1 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=1)  # P_6

        self.conv4_0 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1, stride=2, groups=out_channels)
        self.conv4_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1)  # P_7

        self.add0 = Add()
        self.add1 = Add()
        self.activation = torch.nn.ReLU()

    def forward(self, input):
        base_features = self.get_base_features(input)

        # Coarsest first. C5, C4, C3
        assert [b.shape[2] for b in base_features] == sorted([b.shape[2] for b in base_features])
        assert [b.shape[3] for b in base_features] == sorted([b.shape[3] for b in base_features])

        x = self.conv0_0(base_features[0])  # C5
        p5 = self.conv0_2(self.conv0_1(x))
        x = torch.nn.functional.interpolate(x, base_features[1].size()[2:])  # Upsample 2x

        x = self.add0(x, self.conv1_0(base_features[1]))  # C4
        p4 = self.conv1_2(self.conv1_1(x))
        x = torch.nn.functional.interpolate(x, base_features[2].size()[2:])  # Upsample 2x

        x = self.add1(x, self.conv2_0(base_features[2]))  # C3
        p3 = self.conv2_2(self.conv2_1(x))

        x = self.conv3_0(base_features[0])  # C5
        p6 = self.conv3_1(x)

        x = self.activation(p6)
        p7 = self.conv4_1(self.conv4_0(x))

        return [p3, p4, p5, p6, p7]
