import torch
from .base import ModuleBase


class NonMaxSuppression(ModuleBase):
    def __init__(self, max_detections=10):
        super().__init__()
        self.max_detections = max_detections
        self.iou_threshold = 0.45
        self.prob_threshold = 0.05

    def forward(self, boxes, scores):
        """Non max suppression

        Args:
            boxes: a tensor with shape (num_batches, num_priors, 4)
            scores: a tesnro with shape (num_batches, num_priors, num_classes)

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
        assert len(max_probs.shape) == 1
        assert len(max_classes.shape) == 1
        assert len(max_probs) == len(max_classes)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < self.max_detections:
            # Select the prediction with the highest probability.
            prob, i = torch.max(max_probs, dim=0)
            if prob < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(int(max_classes[i]))
            selected_probs.append(float(max_probs[i]))

            box = boxes[i]
            other_indices = torch.cat((torch.arange(i), torch.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = torch.max(box[0], other_boxes[:, 0])
            y1 = torch.max(box[1], other_boxes[:, 1])
            x2 = torch.min(box[2], other_boxes[:, 2])
            y2 = torch.min(box[3], other_boxes[:, 3])
            w = torch.clamp(x2 - x1, min=0, max=1)
            h = torch.clamp(y2 - y1, min=0, max=1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            # overlapping_indices = torch.squeeze(other_indices[torch.nonzero(iou > self.iou_threshold)])
            overlapping_indices = other_indices[torch.nonzero(iou > self.iou_threshold)].view(-1)
            overlapping_indices = torch.cat((overlapping_indices, torch.tensor([i], device=overlapping_indices.device)))  # Append i to the indices.

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            # This is assuming multi-label predictions.
            scores[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices], max_classes[overlapping_indices] = torch.max(scores[overlapping_indices], dim=1)

        assert len(selected_boxes) == len(selected_classes) and len(selected_boxes) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs
