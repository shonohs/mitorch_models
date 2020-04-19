import torch
from .base import ModuleBase


class ChannelShuffle(ModuleBase):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, input):
        batch_size, channels, height, width = input.data.size()
        x = input.view(-1, self.num_groups, channels // self.num_groups, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch_size, channels, height, width)
