import torch
import torchvision
from .base import ModuleBase


class NonMaxSuppression(ModuleBase):
    def __init__(self, max_detections=10):
        super().__init__()
        self.max_detections = max_detections
        self.iou_threshold = 0.45
        self.prob_threshold = 0.05

    def forward(self, boxes, scores):
        """Non max suppression

        This is class agnostic.

        Args:
            boxes: a tensor with shape (num_batches, num_priors, 4)
            scores: a tensor with shape (num_batches, num_priors, num_classes)

        Returns:
           [[selected_boxes, selected_classes, selected_scores], [selected_boxes, ...], ...]
        """
        assert len(boxes) == len(scores)
        assert len(boxes.shape) == 3 and boxes.shape[2] == 4
        assert len(scores.shape) == 3

        return [self.forward_one_batch(boxes[i], scores[i]) for i in range(len(boxes))]

    def forward_one_batch(self, boxes, scores):
        assert len(boxes) == len(scores)
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4
        assert len(scores.shape) == 2

        # max_probs: shape (num_prior,). max_classes: shape (num_prior,).
        max_probs, max_classes = torch.max(scores, dim=1)

        prob_indices = max_probs >= self.prob_threshold
        boxes = boxes[prob_indices]
        max_probs = max_probs[prob_indices]
        max_classes = max_classes[prob_indices]

        indices = torchvision.ops.nms(boxes, max_probs, self.iou_threshold)

        selected_boxes = boxes[indices]
        selected_classes = max_classes[indices]
        selected_probs = max_probs[indices]

        return selected_boxes, selected_classes, selected_probs
