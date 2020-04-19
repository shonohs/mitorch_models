"""SE-ResNext"""
from .resnext import ResNext
from .modules import SEBlock


class SEResNext(ResNext):
    class BasicBlock(ResNext.BasicBlock):
        def __init__(self, in_channels, out_channels, cardinality, stride=1, reduction_ratio=16):
            super().__init__(in_channels, out_channels, cardinality, stride)
            self.se = SEBlock(out_channels, reduction_ratio)

        def forward(self, input):
            x = self.conv0(input)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.se(x)

            x = self.add(x, self.conv_shortcut(input) if self.conv_shortcut else input)
            return self.activation(x)

    def __init__(self, num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, reduction_ratio=16):
        self.reduction_ratio = reduction_ratio
        super().__init__(num_blocks, cardinality, bottleneck_width)

    def _make_stage(self, in_channels, out_channels, num_blocks, first_block_stride, cardinality, index):
        blocks = [(f'block{index}_0', SEResNext.BasicBlock(in_channels, out_channels, cardinality, first_block_stride, self.reduction_ratio))]
        for i in range(num_blocks - 1):
            blocks.append((f'block{index}_{i+1}', SEResNext.BasicBlock(out_channels, out_channels, cardinality, reduction_ratio=self.reduction_ratio)))
        return blocks


class SEResNext14(SEResNext):
    def __init__(self):
        super().__init__([1, 1, 1, 1])


class SEResNext26(SEResNext):
    def __init__(self):
        super().__init__([2, 2, 2, 2])


class SEResNext50(SEResNext):
    def __init__(self):
        super().__init__([3, 4, 6, 3])


class SEResNext101(SEResNext):
    def __init__(self):
        super().__init__([3, 4, 23, 3])
