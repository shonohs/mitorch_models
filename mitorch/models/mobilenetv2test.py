import torch
import torch.nn as nn
from .classifiertest import Classifier


class MobileNetV2Test(Classifier):
    INPUT_SIZE = 224

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
            super().__init__()
            intermediate_planes = in_planes * expansion_factor
            self.conv0 = nn.Conv2d(in_planes, intermediate_planes, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(intermediate_planes)
            self.conv1 = nn.Conv2d(intermediate_planes, intermediate_planes, kernel_size=3, padding=1, stride=stride, groups=intermediate_planes, bias=False)
            self.bn1 = nn.BatchNorm2d(intermediate_planes)
            self.conv2 = nn.Conv2d(intermediate_planes, out_planes, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=True)
            self.residual = stride == 1 and in_planes == out_planes

        def forward(self, x):
            out = self.relu(self.bn0(self.conv0(x)))
            out = self.relu(self.bn1(self.conv1(out)))
            out = self.bn2(self.conv2(out)) # Without ReLU.
            return out + x if self.residual else out

    def __init__(self, num_classes, width_multiplier = 1):
        super().__init__(num_classes, int(1280 * width_multiplier))
        self.features = nn.Sequential(
            nn.Conv2d(3, int(32 * width_multiplier), kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(int(32 * width_multiplier)),
            nn.ReLU(inplace=True),
            self._make_stage(int(32 * width_multiplier), int(16 * width_multiplier), expansion_factor=1, num_block=1, stride=1),
            self._make_stage(int(16 * width_multiplier), int(24 * width_multiplier), expansion_factor=6, num_block=2, stride=2),
            self._make_stage(int(24 * width_multiplier), int(32 * width_multiplier), expansion_factor=6, num_block=3, stride=2),
            self._make_stage(int(32 * width_multiplier), int(64 * width_multiplier), expansion_factor=6, num_block=4, stride=2),
            self._make_stage(int(64 * width_multiplier), int(96 * width_multiplier), expansion_factor=6, num_block=3, stride=1),
            self._make_stage(int(96 * width_multiplier), int(160 * width_multiplier), expansion_factor=6, num_block=3, stride=2),
            self._make_stage(int(160 * width_multiplier), int(320 * width_multiplier), expansion_factor=6, num_block=1, stride=1),
            nn.Conv2d(int(320 * width_multiplier), int(1280 * width_multiplier), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(1280 * width_multiplier)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))

        self.reset_parameters()

    def _make_stage(self, in_planes, out_planes, num_block, expansion_factor, stride = 1):
        blocks = [MobileNetV2Test.BasicBlock(in_planes, out_planes, expansion_factor, stride)]
        for i in range(num_block-1):
            blocks.append(MobileNetV2Test.BasicBlock(out_planes, out_planes, expansion_factor))
        return nn.Sequential(*blocks)

class MobileNetV2W2(MobileNetV2Test):
    def __init__(self, num_classes):
        super(MobileNetV2W2, self).__init__(num_classes, 2)
