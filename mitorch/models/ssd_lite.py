import torch
from .model import Model
from .modules import Conv2dAct, PriorBox, SSDLoss, SSDPredictor

class SSDLite(Model):
    class DetectionBlock(torch.nn.Module):
        def __init__(self, in_channels, num_outputs, num_classes):
            super(SSDLite.DetectionBlock, self).__init__()
            self.conv_loc0 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            self.conv_loc1 = torch.nn.Conv2d(in_channels, num_outputs * 4, kernel_size=1)

            self.conv_cls0 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            self.conv_cls1 = torch.nn.Conv2d(in_channels, (num_classes + 1) * num_outputs, kernel_size=1)

        def forward(self, input):
            loc = self.conv_loc1(self.conv_loc0(input))
            cls = self.conv_cls1(self.conv_cls0(input))
            return loc, cls

    def __init__(self, backbone, num_classes, prior_box = None):
        super(SSDLite, self).__init__(None, use_bn=True)
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
