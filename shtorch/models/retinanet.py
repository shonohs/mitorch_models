import torch
from .model import Model
from .modules import Conv2dAct, FocalLoss, RetinaPriorBox, RetinaPredictor

class RetinaNet(Model):
    class DetectionBlock(torch.nn.Module):
        def __init__(self, in_channels, num_outputs, num_classes):
            super(RetinaNet.DetectionBlock, self).__init__()

            self.conv_loc0 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_loc1 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_loc2 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_loc3 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_loc4 = torch.nn.Conv2d(in_channels, num_outputs * 4, kernel_size=3, padding=1)

            self.conv_cls0 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_cls1 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_cls2 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_cls3 = Conv2dAct(in_channels, in_channels, kernel_size=3, padding=1)
            self.conv_cls4 = torch.nn.Conv2d(in_channels, num_classes * num_outputs, kernel_size=3, padding=1)

        def forward(self, input):
            loc = self.conv_loc0(input)
            loc = self.conv_loc1(loc)
            loc = self.conv_loc2(loc)
            loc = self.conv_loc3(loc)
            loc = self.conv_loc4(loc)

            cls = self.conv_cls0(input)
            cls = self.conv_cls1(cls)
            cls = self.conv_cls2(cls)
            cls = self.conv_cls3(cls)
            cls = self.conv_cls4(cls)

            return loc, cls

    def __init__(self, backbone, num_classes, prior_box = None):
        super(RetinaNet, self).__init__(None)
        self.base_model = backbone

        if not prior_box:
            prior_box = RetinaPriorBox(len(backbone.output_dim))

        num_priors = prior_box.get_num_priors()

        assert backbone.output_dim.count(backbone.output_dim[0]) == len(backbone.output_dim)
        assert num_priors.count(num_priors[0]) == len(num_priors)
        assert len(backbone.output_dim) == len(num_priors)

        self.detection_block = RetinaNet.DetectionBlock(backbone.output_dim[0], num_priors[0], num_classes)
        self.loss = FocalLoss(num_classes, prior_box)
        self.predictor = RetinaPredictor(num_classes, prior_box)

    def forward(self, input):
        features = self.base_model(input)
        return [self.detection_block(feature) for feature in features]
