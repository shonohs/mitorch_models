"""EfficientDet: Scalable and Efficient Object Detection
"""
import collections
import torch
from .head import Head
from ..model import Model
from ..modules import Add, Conv2dAct


class BidirectionalFeaturePyramidNetwork(Head):
    """BiFPN implementation.
       Note that this implementation skipped weighted feature fusion.
    """
    class BasicBlock(Model):
        def __init__(self, input_channels, out_channels):
            super().__init__([out_channels] * 5)

            assert len(input_channels) == 5

            self.conv0_0 = torch.nn.Conv2d(input_channels[0], out_channels, kernel_size=1)  # P_7
            self.conv0_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)  # N_7

            self.conv1_0 = torch.nn.Conv2d(input_channels[1], out_channels, kernel_size=1)  # P_6
            self.conv1_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv1_2 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)  # N_6

            self.conv2_0 = torch.nn.Conv2d(input_channels[2], out_channels, kernel_size=1)  # P_5
            self.conv2_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv2_2 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)  # N_5

            self.conv3_0 = torch.nn.Conv2d(input_channels[3], out_channels, kernel_size=1)  # P_4
            self.conv3_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv3_2 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)  # N_4

            self.conv4_0 = torch.nn.Conv2d(input_channels[4], out_channels, kernel_size=1)  # P_3
            self.conv4_1 = Conv2dAct(out_channels, out_channels, kernel_size=3, padding=1)  # N_3

            self.add = Add()

        def forward(self, input):
            # TODO: Implement weighted feature fusion.
            assert len(input) == 5

            p7 = self.conv0_0(input[0])
            p6 = self.conv1_0(input[1])
            p5 = self.conv2_0(input[2])
            p4 = self.conv3_0(input[3])
            p3 = self.conv4_0(input[4])

            n6 = self.add(p6, torch.nn.functional.interpolate(p7, size=p6.shape[2:]))  # Upsample 2x
            n6 = self.conv1_1(n6)

            n5 = self.add(p5, torch.nn.functional.interpolate(n6, size=p5.shape[2:]))  # Upsample 2x
            n5 = self.conv2_1(n5)

            n4 = self.add(p4, torch.nn.functional.interpolate(n5, size=p4.shape[2:]))  # Upsample 2x
            n4 = self.conv3_1(n4)

            n3 = self.add(p3, torch.nn.functional.interpolate(n4, size=p3.shape[2:]))  # Upsample 2x
            n3 = self.conv4_1(n3)

            n4 = self.add(p4, n4, torch.nn.functional.interpolate(n3, size=p4.shape[2:]))  # Downsample 2x
            n4 = self.conv3_2(n4)

            n5 = self.add(p5, n5, torch.nn.functional.interpolate(n4, size=p5.shape[2:]))  # Downsample 2x
            n5 = self.conv2_2(n5)

            n6 = self.add(p6, n6, torch.nn.functional.interpolate(n5, size=p6.shape[2:]))  # Downsample 2x
            n6 = self.conv1_2(n6)

            n7 = self.add(p7, torch.nn.functional.interpolate(n6, size=p7.shape[2:]))  # Downsample 2x
            n7 = self.conv0_1(n7)

            return [n7, n6, n5, n4, n3]

    def __init__(self, backbone, out_channels=256, num_blocks=1):
        super().__init__(backbone, [5, 4, 3], [out_channels] * 5)
        base_output_shapes = Head.get_base_output_shapes(backbone, [5, 4, 3])

        self.base_model = backbone

        self.conv0 = torch.nn.Conv2d(base_output_shapes[0], out_channels, kernel_size=3, padding=1, stride=2)  # P_6
        self.conv1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)  # P_7
        self.activation = torch.nn.ReLU()

        basic_blocks = []
        input_channels = [out_channels, out_channels, base_output_shapes[0], base_output_shapes[1], base_output_shapes[2]]
        for i in range(num_blocks):
            basic_blocks.append((f'block{i}', BidirectionalFeaturePyramidNetwork.BasicBlock(input_channels, out_channels)))
            input_channels = [out_channels] * 5

        self.basic_blocks = torch.nn.Sequential(collections.OrderedDict(basic_blocks))

    def forward(self, input):
        base_features = self.get_base_features(input)

        # Coarsest first. C5, C4, C3
        assert [b.shape[2] for b in base_features] == sorted([b.shape[2] for b in base_features])
        assert [b.shape[3] for b in base_features] == sorted([b.shape[3] for b in base_features])

        c5, c4, c3 = base_features

        p6 = self.conv0(c5)
        p7 = self.conv1(self.activation(self.conv1(p6)))

        n7, n6, n5, n4, n3 = self.basic_blocks((p7, p6, c5, c4, c3))
        return [n3, n4, n5, n6, n7]
