import torch
from .model import Model
from .modules import YoloLoss, YoloPredictor, PriorBox


class YoloV2(Model):
    def __init__(self, backbone, num_classes, prior_box=None):
        super().__init__(None)
        self.base_model = backbone

        if not prior_box:
            prior_box = PriorBox(len(backbone.output_dim))

        num_priors = prior_box.get_num_priors()[0]
        self.detection_block = torch.nn.Conv2d(backbone.output_dim[0], (num_classes + 5) * num_priors, kernel_size=1)
        self.loss = YoloLoss(num_classes, prior_box)
        self.predictor = YoloPredictor(num_classes, prior_box)

    def forward(self, input_tensor):
        features = self.base_model(input_tensor)
        return self.detection_block(features)
