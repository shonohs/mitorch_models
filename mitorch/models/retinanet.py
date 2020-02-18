import collections
import math
import torch
from .model import Model
from .modules import Conv2dAct, FocalLoss, RetinaPriorBox, RetinaPredictor, ModuleBase

class RetinaNet(Model):
    class DetectionBlock(ModuleBase):
        def __init__(self, in_channels, num_outputs, num_classes, num_blocks):
            super(RetinaNet.DetectionBlock, self).__init__()
            self.conv_loc = torch.nn.Sequential(collections.OrderedDict(
                [(f'conv{i}', Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)) for i in range(num_blocks)]
                + [(f'conv{num_blocks}', torch.nn.Conv2d(in_channels, num_outputs * 4, kernel_size=3, padding=1))]
            ))

            self.conv_cls = torch.nn.Sequential(collections.OrderedDict(
                [(f'conv{i}', Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)) for i in range(num_blocks)]
                + [(f'conv{num_blocks}', torch.nn.Conv2d(in_channels, num_classes * num_outputs, kernel_size=3, padding=1))]
            ))

        def forward(self, input):
            return self.conv_loc(input), self.conv_cls(input)

        def reset_parameters(self):
            pi = 0.01
            self.conv_cls[-1].bias.data.fill_(-math.log((1-pi)/pi))

    def __init__(self, backbone, num_classes, prior_box = None, num_blocks = 4):
        super(RetinaNet, self).__init__(None)
        self.base_model = backbone

        if not prior_box:
            prior_box = RetinaPriorBox(len(backbone.output_dim))

        num_priors = prior_box.get_num_priors()

        assert backbone.output_dim.count(backbone.output_dim[0]) == len(backbone.output_dim)
        assert num_priors.count(num_priors[0]) == len(num_priors)
        assert len(backbone.output_dim) == len(num_priors)

        self.detection_block = RetinaNet.DetectionBlock(backbone.output_dim[0], num_priors[0], num_classes, num_blocks)
        self.loss = FocalLoss(num_classes, prior_box)
        self.predictor = RetinaPredictor(num_classes, prior_box)

    def forward(self, input):
        features = self.base_model(input)

        assert [f.shape[2] for f in features] == sorted([f.shape[2] for f in features], reverse=True)
        return [self.detection_block(feature) for feature in features]
