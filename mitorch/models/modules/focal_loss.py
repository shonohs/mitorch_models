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

        valid_indices = target_classification >= 0
        pred_classification = pred_classification[valid_indices]
        target_classification = target_classification[valid_indices]

        if len(target_classification) == 0:
            return torch.tensor(0, dtype=torch.float, requires_grad=True)

        target = FocalLoss._get_one_hot(target_classification, self.num_classes,
                                        pred_classification.dtype,
                                        pred_classification.layout,
                                        pred_classification.device)

        assert pred_classification.shape == target.shape

        # If y == 0, l = -log(1-sigmoid(x)). if y == 1, l = -log(sigmoid(x))
        ce = torch.nn.functional.binary_cross_entropy_with_logits(pred_classification, target, reduction='none')
        pt = torch.exp(-ce)  # if y == 0, pt = 1 - sigmoid(x). if y == 1, pt = sigmoid(x)

        assert ce.shape[0] == target.shape[0] and ce.shape[1] == target.shape[1]
        assert pt.shape[0] == target.shape[0] and pt.shape[1] == target.shape[1]

        return (ce * self.alpha * ((1 - pt) ** self.gamma)).sum()
