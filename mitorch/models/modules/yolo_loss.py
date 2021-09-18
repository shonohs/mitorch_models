"""
You Only Look Once: Unified, Real-Time Object Detection (https://arxiv.org/pdf/1506.02640.pdf)
YOLO9000: Better, Faster, Stronger (https://arxiv.org/pdf/1612.08242.pdf)
"""

import torch
from .base import ModuleBase


class YoloLoss(ModuleBase):
    def __init__(self, num_classes, prior_box, object_scale=5):
        super().__init__()
        self.num_classes = num_classes
        self.prior_box = prior_box
        self.object_scale = 5
        self.negative_location_scale = 0.01

    def encode_target(self, prediction, targets, priors):
        """

            Find one bounding box that matches the ground truth, then use it as positive.
        """
        target_tensors = [torch.tensor(target, dtype=torch.float32, device=priors.device) for target in targets]
        results = [self.encode_one_batch(prediction[i], target_tensors[i], priors) for i in range(len(prediction))]
        return torch.stack(results)

    def encode_one_batch(self, prediction, target, priors):
        if target.shape[0] == 0:
            return torch.zeros_like(prediction)

        ious = self.iou(target[:, 1:], priors)

        best_prior_overlap, best_prior_index = ious.max(1)  # Shape: (num_labels,)
        matched_priors = priors[best_prior_index]
        matched_priors_center_xy = (matched_priors[:, :2] + matched_priors[:, 2:4]) / 2
        matched_priors_wh = matched_priors[:, 2:4] - matched_priors[:, :2]

        target_center_xy = (target[:, 1:3] + target[:, 3:5]) / 2
        target_wh = target[:, 3:5] - target[:, 1:3]

        encoded = torch.zeros_like(prediction)
        encoded[best_prior_index, :2] = target_center_xy - matched_priors_center_xy
        encoded[best_prior_index, 2:4] = torch.log(target_wh / matched_priors_wh)
        encoded[best_prior_index, 4] = 1

        class_index = target[:, 0:1].long()  # Shape: (num_labels, 1)
        encoded[best_prior_index, 5:] = encoded[best_prior_index, 5:].scatter(1, class_index, 1)
        return encoded

    def forward(self, predictions, targets):
        assert len(predictions.shape) == 4 and predictions.shape[1] % (self.num_classes + 5) == 0
        priors = self.prior_box(predictions)

        ps = predictions.shape
        predictions = predictions.view(ps[0], -1, self.num_classes + 5, ps[2], ps[3]).permute(0, 3, 4, 1, 2).view(ps[0], -1, self.num_classes + 5)

        # The third dimension is x, y, w, h, confidence, class0, class1, ...
        encoded_targets = self.encode_target(predictions, targets, priors)
        object_mask = encoded_targets[:, :, 4] > 0

        masked_predictions = predictions[object_mask]
        masked_targets = encoded_targets[object_mask]

        # Skip objectness loss if the priors have 0.6 IOU with targets.

        location_loss = torch.nn.functional.mse_loss(masked_predictions[:, :, :4, :, :], masked_targets[:, :, :4, :, :], reduction='sum')
        object_loss = torch.nn.functional.mse_loss(torch.sigmoid(masked_predictions[:, :, 4, :, :]), masked_targets[:, :, 4, :, :], reduction='sum')
        no_object_loss = torch.nn.functional.mse_loss(torch.sigmoid(predictions[not object_mask][:, :, 4, :, :]), encoded_targets[not object_mask][:, :, 4, :, :], reduction='sum')

        # TODO: the official code uses softmax
        class_loss = torch.nn.functional.mse_loss(torch.sigmoid(masked_predictions[:, :, 5:, :, :]), masked_targets[:, :, 5:, :, :], reduction='sum')

        return location_loss + object_loss * self.object_scale + no_object_loss + class_loss

    def iou(self, box0, box1):
        """ Compute Intersecion Over Union (IOU) between given two set of boxes.
        Args:
            box0 (N0, 4)
            box1 (N1, 4)
        Return: iou (N0, N1)
        """
        # Get Intersection
        max_xy = torch.min(box0[:, 2:].unsqueeze(1).expand(box0.size(0), box1.size(0), 2), box1[:, 2:].unsqueeze(0).expand(box0.size(0), box1.size(0), 2))
        min_xy = torch.max(box0[:, :2].unsqueeze(1).expand(box0.size(0), box1.size(0), 2), box1[:, :2].unsqueeze(0).expand(box0.size(0), box1.size(0), 2))
        intersection = torch.clamp((max_xy - min_xy), min=0)

        intersection_areas = intersection[:, :, 0] * intersection[:, :, 1]

        box0_areas = ((box0[:, 2] - box0[:, 0]) * (box0[:, 3] - box0[:, 1])).unsqueeze(1).expand_as(intersection_areas)
        box1_areas = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(0).expand_as(intersection_areas)
        union_areas = box0_areas + box1_areas - intersection_areas
        return intersection_areas / union_areas
