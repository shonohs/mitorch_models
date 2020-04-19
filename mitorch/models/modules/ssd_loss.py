import math
import torch
from .base import ModuleBase


class SSDLoss(ModuleBase):
    def __init__(self, num_classes, prior_box):
        super().__init__()
        self.negative_iou_threshold = 0.5
        self.positive_iou_threshold = 0.5
        self.neg_pos_ratio = 3
        self.num_classes = num_classes
        self.prior_box = prior_box
        self.num_classifier = num_classes + 1  # SSD has background class.

    def hard_negative_mining(self, pred_classification, target_classification, neg_pos_ratio):
        """ Hard negative mining. Returns the indices for the selected entries.
        Args:
            pred_classification (N, num_prior, num_class+1)
            target_classification (N, num_prior)
        """
        assert len(pred_classification.shape) == 3 and pred_classification.shape[2] == self.num_classes + 1
        assert len(target_classification.shape) == 2

        pos_mask = target_classification > 0
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * neg_pos_ratio

        with torch.no_grad():
            negative_scores = torch.nn.functional.log_softmax(pred_classification, dim=2)[:, :, 0]  # (N, num_prior)
            negative_scores[pos_mask] = math.inf  # Exclude positive boxes.

            # Get Top-k mask
            _, indexes = negative_scores.sort(dim=1)
            _, orders = indexes.sort(dim=1)
            neg_mask = orders < num_neg

            return pos_mask | neg_mask

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

    def assign_priors(self, target, priors, negative_iou_threshold, positive_iou_threshold):
        """Given ground truth boxes, find corresponding prior boxes and calculate location diffs and labels for SSD loss.
        Args:
            target: shape (num_labels, 5) tensor [label, min_x, min_y, max_x, max_y]
            priors: (num_priors, 4) [min_x, min_y, max_x, max_y]
            negative_iou_threshold: maxmimum IOU between a ground truth and a prior box to be considered as negative example.
            positive_iou_threshold: minimum IOU between a ground truth and a prior box to be considiered as matched.
        Returns: (target_location, target_labels)
            target_location (num_priors, 4)
            target_labels (num_priors,)
        """
        assert negative_iou_threshold <= positive_iou_threshold

        # If there is no golden truths, return an empty array.
        if target.shape[0] == 0:
            return torch.zeros_like(priors), torch.zeros(priors.shape[0], dtype=torch.long, device=target.device)

        assert len(target.size()) == 2 and target.size()[1] == 5, "Target size is {}".format(target.size())

        # Get overlaps between target and prior bounding boxes.
        # ious: shape (num_labels, num_priors)
        ious = self.iou(target[:, 1:], priors)

        # Best matching prior for each ground truth
        best_prior_overlap, best_prior_index = ious.max(1)  # Shape: (num_labels,)
        assert len(best_prior_overlap.shape) == 1 and best_prior_overlap.shape[0] == target.shape[0]

        # Best matching ground truth for each prior
        best_target_overlap, best_target_index = ious.max(0)  # Shape: (num_priors,)
        assert len(best_target_overlap.shape) == 1 and best_target_overlap.shape[0] == priors.shape[0]

        # Make sure the best_prior_index will be bigger than the threshold. The threshold should be less than 1.0.
        best_target_overlap.index_fill_(0, best_prior_index, 1.0)
        best_target_index[best_prior_index] = torch.tensor(range(len(best_prior_index)), device=target.device)

        matched_targets = target[best_target_index, :]  # Shape: (num_priors, 5)
        target_labels = matched_targets[:, 0] + 1  # Increment to make the label 0 background.
        target_labels[best_target_overlap < negative_iou_threshold] = 0  # If IOU is too small, set it as background.
        # If IOU is between negative_iou_threshold and positive_iou_threshold, ignore it.
        target_labels[(best_target_overlap >= negative_iou_threshold) & (best_target_overlap < positive_iou_threshold)] = -1

        priors_center_xy = (priors[:, :2] + priors[:, 2:]) / 2
        priors_wh = priors[:, 2:] - priors[:, :2]

        # Find the difference between prior and matched ground truths.
        # If there is no matched ground truths, the result is random.
        diff_center_xy = (matched_targets[:, 1:3] + matched_targets[:, 3:]) / 2 - priors_center_xy
        diff_center_xy /= priors_wh * 0.1  # Variance: 0.1
        diff_wh = (matched_targets[:, 3:] - matched_targets[:, 1:3]) / priors_wh
        diff_wh = torch.log(diff_wh) / 0.2  # Variance: 0.2

        target_location = torch.cat([diff_center_xy, diff_wh], 1)  # Shape: (N, num_priors, 4)

        assert target_location.size()[0] == target_labels.size()[0] and target_location.size()[1] == 4

        return target_location, target_labels.long()

    def encode_target(self, targets, priors):
        """ Encode a list of ground truths into the compatible format with the SSD output.
        input:
            targets: List of ground truths. (N, num_labels, 5)

        returns: (target_location, target_classification)
        """
        target_locations = []
        target_classifications = []
        for target in targets:
            target = torch.tensor(target, device=priors.device)
            target_location, target_classification = self.assign_priors(target, priors, self.negative_iou_threshold, self.positive_iou_threshold)
            target_locations.append(target_location)
            target_classifications.append(target_classification)
        return (torch.stack(target_locations), torch.stack(target_classifications))

    def loss_classification(self, pred_classification, target_classification):
        # Hard negative mining
        mask = self.hard_negative_mining(pred_classification, target_classification, self.neg_pos_ratio)  # Shape: (N, num_prior)
        # Use 'sum' reduction since we divide the loss by num_positive later.
        loss_classification = torch.nn.functional.cross_entropy(pred_classification[mask], target_classification[mask], reduction='sum')
        return loss_classification

    def reshape_ssd_predictions(self, predictions):
        num_batch = len(predictions[0][0])

        pred_location = []
        pred_classification = []
        for loc, cls in predictions:
            loc = loc.permute(0, 2, 3, 1).contiguous().view(num_batch, -1, 4)
            cls = cls.permute(0, 2, 3, 1).contiguous().view(num_batch, -1, self.num_classifier)
            pred_location.append(loc)
            pred_classification.append(cls)
        return (torch.cat(pred_location, dim=1), torch.cat(pred_classification, dim=1))

    def forward(self, predictions, targets):
        """
        predictions: [(pred_loc, pred_cls), (pred_loc, pred_clas), ...],
                     where pred_loc is (N, num_priors*4, W, H) and pred_cls is (N, num_priors*(num_classes+1), W, H)
                     Note that pred_cls's 0th output is for negative prediction classes.

        target: (N, num_label, 5)
            e.g. [[(label, x0, y0, x1, y1), ...], [...], ...]
        """

        pred_location, pred_classification = self.reshape_ssd_predictions(predictions)
        priors = self.prior_box(predictions)

        batch_num = len(pred_location)

        # pred_location.shape should be (N, num_prior, 4)
        assert (len(pred_location.shape) == 3 and pred_location.shape[0] == batch_num
                and pred_location.shape[1] == len(priors) and pred_location.shape[2] == 4), "pred_location shape {} is not expected".format(pred_location.shape)

        # pred_classification.shape should be (N, num_prior, num_classes + 1)
        assert (len(pred_classification.shape) == 3 and pred_classification.shape[0] == batch_num
                and pred_classification.shape[1] == len(priors) and pred_classification.shape[2] == self.num_classifier)

        # target_location: shape (N, num_prior, 4)
        # target_classification: shape (N, num_prior)
        target_location, target_classification = self.encode_target(targets, priors)
        assert len(target_location.shape) == 3 and target_location.shape[2] == 4
        assert len(target_classification.shape) == 2 and target_classification.shape[0] == batch_num

        positive_priors_index = target_classification > 0  # Shape: (N, num_prior)
        loss_location = torch.nn.functional.smooth_l1_loss(pred_location[positive_priors_index].view(-1, 4), target_location[positive_priors_index].view(-1, 4), reduction='sum')
        loss_classification = self.loss_classification(pred_classification, target_classification)

        num_positive = positive_priors_index.long().sum()

        loss = (loss_location + loss_classification) / num_positive
        return loss
