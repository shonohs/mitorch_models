import math
import torch
from .model import Model
from .modules import DepthwiseSeparableConv2d, PriorBox, SSDLoss, SSDSigmoidLoss, SSDPredictor, SSDSigmoidPredictor, default_module_settings, ModuleBase


class SSDLite(Model):
    class DetectionBlock(ModuleBase):
        def __init__(self, in_channels, num_outputs, num_classifiers):
            super().__init__()
            self.conv_loc = DepthwiseSeparableConv2d(in_channels, num_outputs * 4, kernel_size=3, padding=1, use_bn2=False, activation2='none')
            self.conv_cls = DepthwiseSeparableConv2d(in_channels, num_classifiers * num_outputs, kernel_size=3, padding=1, use_bn2=False, activation2='none')

        def forward(self, input):
            loc = self.conv_loc(input)
            cls = self.conv_cls(input)
            return loc, cls

        def reset_parameters(self):
            self.conv_loc.reset_parameters()
            self.conv_cls.reset_parameters()

            # Borrowed from RetinNet. TODO: Do we have an evidence that this works for SSDLite?
            pi = 0.01
            self.conv_cls.pointwise_conv.conv.bias.data.fill_(-math.log((1 - pi) / pi))

    @default_module_settings(use_bn=True)
    def __init__(self, backbone, num_classes, prior_box=None, use_sigmoid=False):
        super().__init__(None)
        self.base_model = backbone

        if not prior_box:
            prior_box = PriorBox(len(backbone.output_dim))

        num_priors = prior_box.get_num_priors()
        num_classifiers = num_classes if use_sigmoid else (num_classes + 1)
        self.detection_blocks = torch.nn.ModuleList([SSDLite.DetectionBlock(dim, num, num_classifiers) for dim, num in zip(backbone.output_dim, num_priors)])
        self.loss = (SSDSigmoidLoss if use_sigmoid else SSDLoss)(num_classes, prior_box)
        self.predictor = (SSDSigmoidPredictor if use_sigmoid else SSDPredictor)(num_classes, prior_box)

    def forward(self, input):
        features = self.base_model(input)
        assert len(features) == len(self.detection_blocks)
        return [b(features[i]) for i, b in enumerate(self.detection_blocks)]

    def reset_parameters(self):
        for b in self.detection_blocks:
            b.reset_parameters()
