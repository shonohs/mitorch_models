"""
"Focal Loss for Dense Object Detection" (https://arxiv.org/pdf/1708.02002.pdf)
"""
import torch
from .ssd_loss import SSDSigmoidLoss


class FocalLoss(SSDSigmoidLoss):
    def __init__(self, num_classes, prior_box, alpha=0.25, gamma=2):
        super().__init__(num_classes, prior_box)
        self.negative_iou_threshold = 0.4
        self.positive_iou_threshold = 0.5
        self.alpha = alpha
        self.gamma = gamma

    def loss_classification(self, pred_classification, target_classification):
        """Classification loss
        Args:
            pred_classification: (N, num_priors, num_classes)
            target_classification: (N, num_priors)

        Note that for the target classification, "0" represents the background.
        """
        assert len(pred_classification.shape) == 3 and pred_classification.shape[2] == self.num_classifiers
        assert len(target_classification.shape) == 2 and pred_classification.shape[0:2] == target_classification.shape[0:2]

        pred_classification = pred_classification.view(-1, self.num_classes)
        target_classification = target_classification.view(-1)

        # Ignore priors that are neither foreground or background.
        # Note that it's guaranteed to have at least one foreground target.
        valid_indexes = target_classification >= 0
        pred_classification = pred_classification[valid_indexes]
        target_classification = target_classification[valid_indexes]

        target = FocalLoss._get_one_hot(target_classification, self.num_classes, pred_classification.dtype, pred_classification.layout, pred_classification.device)
        assert pred_classification.shape == target.shape

        # If y == 0, l = -log(1-sigmoid(x)). if y == 1, l = -log(sigmoid(x))
        ce = torch.nn.functional.binary_cross_entropy_with_logits(pred_classification, target, reduction='none')

        p = torch.sigmoid(pred_classification)
        pt = p * target + (1 - p) * (1 - target)  # if y == 0, pt = 1 - sigmoid(x). if y == 1, pt = sigmoid(x)
        pt = pt.detach()

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        assert ce.shape == pt.shape == alpha_t.shape

        # Quote from the paper: "We perform the normalization by the number of assigned anvhors, not total anchors, ..."
        return (ce * alpha_t * ((1 - pt) ** self.gamma)).sum()
