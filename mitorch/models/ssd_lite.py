import torch
from .model import Model
from .modules import DepthwiseSeparableConv2d, PriorBox, SSDLoss, SSDPredictor, default_module_settings


class SSDLite(Model):
    class DetectionBlock(torch.nn.Module):
        def __init__(self, in_channels, num_outputs, num_classes):
            super().__init__()
            self.conv_loc = DepthwiseSeparableConv2d(in_channels, num_outputs * 4, kernel_size=3, padding=1, use_bn2=False, activation2='none')
            self.conv_cls = DepthwiseSeparableConv2d(in_channels, (num_classes + 1) * num_outputs, kernel_size=3, padding=1, use_bn2=False, activation2='none')

        def forward(self, input):
            loc = self.conv_loc(input)
            cls = self.conv_cls(input)
            return loc, cls

    @default_module_settings(use_bn=True)
    def __init__(self, backbone, num_classes, prior_box=None):
        super().__init__(None)
        self.base_model = backbone

        if not prior_box:
            prior_box = PriorBox(len(backbone.output_dim))

        num_priors = prior_box.get_num_priors()

        self.detection_blocks = torch.nn.ModuleList([SSDLite.DetectionBlock(dim, num, num_classes) for dim, num in zip(backbone.output_dim, num_priors)])
        self.loss = SSDLoss(num_classes, prior_box)
        self.predictor = SSDPredictor(num_classes, prior_box)

    def forward(self, input):
        features = self.base_model(input)
        assert len(features) == len(self.detection_blocks)
        return [b(features[i]) for i, b in enumerate(self.detection_blocks)]
