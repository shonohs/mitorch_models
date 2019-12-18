import torch
from .ssd_loss import SSDLoss


class FocalLoss(SSDLoss):
    def __init__(self, num_classes, prior_box, alpha = 0.25, gamma = 2):
        super(FocalLoss, self).__init__(num_classes, prior_box)
        self.num_classes
        self.num_classifier = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def loss_classification(self, pred_classification, target_classification):
        """Classification loss
        Args:
            pred_classification: (N, num_priors, num_classes)
            target_classification: (N, num_priors)
        """
        assert len(pred_classification.shape) == 3 and pred_classification.shape[2] == self.num_classes
        assert len(target_classification.shape) == 2 and pred_classification.shape[0:2] == target_classification.shape[0:2]

        # Multi-label
        target = torch.zeros_like(pred_classification)
        target.scatter_(2, target_classification.unsqueeze(-1), 1)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(pred_classification, target, reduction='none')
        pt = torch.exp(-ce)
        return (ce * self.alpha * ((1 - pt) ** self.gamma)).sum()
