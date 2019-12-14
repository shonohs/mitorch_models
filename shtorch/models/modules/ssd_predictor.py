import torch
import numpy as np
from .base import ModuleBase
from .non_max_suppression import NonMaxSuppression


class SSDPredictor(ModuleBase):
    def __init__(self, num_classes, prior_box):
        super(SSDPredictor, self).__init__()
        self.num_classes = num_classes
        self.prior_box = prior_box

        self.max_detections = 10
        self.iou_threshold = 0.45
        self.prob_threshold = 0.05

        self.non_max_suppression = NonMaxSuppression()

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes
        Args:
            boxes: shape (num_prior, 4)
            class_probs: shape (num_prior, num_class)
        """


    def reshape_ssd_predictions(self, predictions):
        num_batch = len(predictions[0][0])

        pred_location = []
        pred_classification = []
        for loc, cls in predictions:
            loc = loc.permute(0, 2, 3, 1).contiguous().view(num_batch, -1, 4)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(num_batch, -1, self.num_classes + 1)
            pred_location.append(loc)
            pred_classification.append(cls)
        return (torch.cat(pred_location, dim=1), torch.cat(pred_classification, dim=1))

    def forward(self, predictions):
        """ Get Bounding boxes.
        Args:
            predictions: (location_output, classification_output)
        Returns: (result_boxes, result_classes, result_probs)
            result_boxes: Size N List of List of size 4 tensor.
            result_classes: Size N List of List.
            result_probs: Size N List of List.
        """
        pred_location, pred_classification = self.reshape_ssd_predictions(predictions)
        prior_box = self.prior_box(predictions)

        batch_num = len(pred_location)

        prior_centers = (prior_box[:, :2] + prior_box[:, 2:]) / 2
        prior_sizes = prior_box[:, 2:] - prior_box[:, :2]

        pred_location_centers = prior_centers + pred_location[:,:,:2] * prior_sizes * 0.1 # Variance: 0.1
        pred_location_sizes = prior_sizes * torch.exp(pred_location[:,:,2:] * 0.2) # Variance: 0.2

        assert len(pred_location_centers.shape) == 3 and pred_location_centers.shape[2] == 2
        assert len(pred_location_sizes.shape) == 3 and pred_location_sizes.shape[2] == 2

        # Shape: (N, num_prior, 4)
        bounding_boxes = torch.cat((pred_location_centers - pred_location_sizes / 2, pred_location_centers + pred_location_sizes / 2), 2)
        assert len(bounding_boxes.shape) == 3 and bounding_boxes.shape[2] == 4

        # Remove background predictions and adjust the class number
        pred_classification = torch.nn.functional.softmax(pred_classification, dim=-1)
        pred_classification = pred_classification[:,:,1:] # Shape: (N, num_prior, num_classes)

        results = self.non_max_suppression(bounding_boxes, pred_classification)
        final_results = []
        for boxes, classes, probs in results:
            result = []
            for i in range(len(boxes)):
                result.append([int(classes[i]), float(probs[i]), float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])])
            final_results.append(result)
        return final_results
