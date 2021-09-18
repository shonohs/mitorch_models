import torch
from .base import ModuleBase
from .non_max_suppression import NonMaxSuppression


class YoloPredictor(ModuleBase):
    def __init__(self, num_classes, prior_box):
        super().__init__()
        self.num_classes = num_classes
        self.prior_box = prior_box
        self.non_max_suppression = NonMaxSuppression()

    def forward(self, predictions):
        prior_box = self.prior_box(predictions)

        prior_centers = (prior_box[:, :2] + prior_box[:, 2:]) / 2
        prior_sizes = prior_box[:, 2:] - prior_box[:, :2]

        ps = predictions.shape
        predictions = predictions.view(ps[0], -1, self.num_classes + 5, ps[2], ps[3]).permute(0, 3, 4, 1, 2).view(ps[0], -1, self.num_classes + 5)

        pred_centers = prior_centers + torch.sigmoid(predictions[:, :, :2])
        pred_sizes = prior_sizes * torch.exp(predictions[:, :, 2:4])

        boxes = torch.cat((pred_centers - pred_sizes / 2, pred_centers + pred_sizes / 2), 2)
        assert len(boxes.shape) == 3 and boxes.shape[2] == 4

        pred_classes = predictions[:, :, 4] * torch.sigmoid(predictions[:, :, 5:])

        results = self.non_max_suppression(boxes, pred_classes)
        final_results = []
        for boxes, classes, probs in results:
            final_results.append([[int(classes[i]), float(probs[i]), float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])] for i in range(len(boxes))])
        return final_results
