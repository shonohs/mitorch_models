import torch
from .ssd_predictor import SSDPredictor


class RetinaPredictor(SSDPredictor):
    def __init__(self, num_classes, prior_box):
        super().__init__(num_classes, prior_box)
        self.num_classifier = num_classes

    def predict_class(self, pred_classification):
        assert len(pred_classification.shape) == 3 and pred_classification.shape[2] == self.num_classifier

        return torch.sigmoid(pred_classification)
