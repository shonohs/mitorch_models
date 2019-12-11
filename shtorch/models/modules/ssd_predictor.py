import torch
import numpy as np
from .base import ModuleBase

class SSDPredictor(ModuleBase):
    def __init__(self, num_classes, prior_box):
        super(SSDPredictor, self).__init__()
        self.num_classes = num_classes
        self.prior_box = prior_box

        self.max_detections = 10
        self.iou_threshold = 0.45
        self.prob_threshold = 0.05

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes
        Args:
            boxes: shape (num_prior, 4)
            class_probs: shape (num_prior, num_class)
        """
        assert len(boxes) == len(class_probs)
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4
        assert len(class_probs.shape) == 2 and class_probs.shape[1] == self.num_classes

        # max_probs: shape (num_prior,). max_classes: shape (num_prior,).
        max_probs, max_classes = torch.max(class_probs, dim=1)
        assert len(max_probs.shape) == 1
        assert len(max_classes.shape) == 1

        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            # Select the prediction with the highest probability.
            prob, i = torch.max(max_probs, dim=0)
            if prob < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(int(max_classes[i]))
            selected_probs.append(float(max_probs[i]))

            box = boxes[i]
            other_indices = torch.cat((torch.arange(i), torch.arange(i+1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = torch.max(box[0], other_boxes[:,0])
            y1 = torch.max(box[1], other_boxes[:,1])
            x2 = torch.min(box[2], other_boxes[:,2])
            y2 = torch.min(box[3], other_boxes[:,3])
            w = torch.clamp(x2 - x1, min=0, max=1)
            h = torch.clamp(y2 - y1, min=0, max=1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            #overlapping_indices = torch.squeeze(other_indices[torch.nonzero(iou > self.iou_threshold)])
            overlapping_indices = other_indices[torch.nonzero(iou > self.iou_threshold)].view(-1)
            overlapping_indices = torch.cat((overlapping_indices, torch.tensor([i], device=overlapping_indices.device))) # Append i to the indices.

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            # This is assuming multi-label predictions.
            class_probs[overlapping_indices,max_classes[i]] = 0
            max_probs[overlapping_indices], max_classes[overlapping_indices] = torch.max(class_probs[overlapping_indices], dim=1)

        assert len(selected_boxes) == len(selected_classes) and len(selected_boxes) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs


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

        results = []
        for i in range(batch_num):
            boxes, classes, probs = self._non_maximum_suppression(bounding_boxes[i], pred_classification[i], self.max_detections)
            result = []
            for i in range(len(boxes)):
                result.append([int(classes[i]), float(probs[i]), float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])])
            results.append(result)

        return results
