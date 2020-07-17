import unittest
from unittest.mock import MagicMock
import torch
from mitorch.models.modules import SSDLoss, SSDSigmoidLoss


class TestSSDLoss(unittest.TestCase):
    def test_iou_threshold(self):
        ssd_loss = SSDLoss(5, None)

        target = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.float)
        priors = torch.tensor([[0, 0, 1, 0.1],
                               [0, 0, 1, 0.4],
                               [0, 0, 1, 0.45],
                               [0, 0, 1, 0.5],
                               [0, 0, 1, 1.0]], dtype=torch.float)

        target_loc, target_labels = ssd_loss.assign_priors(target, priors, 0.5, 0.5)
        self.assertEqual(list(target_labels), [0, 0, 0, 1, 1])

        target_loc, target_labels = ssd_loss.assign_priors(target, priors, 0.4, 0.5)
        self.assertEqual(list(target_labels), [0, -1, -1, 1, 1])

    def test_float_double_mixed(self):
        ssd_loss = SSDLoss(5, None)
        target = [torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.double)]
        priors = torch.tensor([[0, 0, 1, 0.1],
                               [0, 0, 1, 0.4],
                               [0, 0, 1, 0.45],
                               [0, 0, 1, 0.5],
                               [0, 0, 1, 1.0]], dtype=torch.float)

        target_loc, target_labels = ssd_loss.encode_target(target, priors)
        self.assertEqual(list(target_labels[0]), [0, 0, 0, 1, 1])

    def test_encode_target_with_no_target(self):
        ssd_loss = SSDLoss(5, None)
        target = [torch.tensor([[]], dtype=torch.float).reshape(0, 5), torch.tensor([[]], dtype=torch.float).reshape(0, 5)]
        priors = torch.tensor([[0, 0, 1, 0.1],
                               [0, 0, 1, 0.4],
                               [0, 0, 1, 0.45],
                               [0, 0, 1, 0.5],
                               [0, 0, 1, 1.0]], dtype=torch.float)

        target_loc, target_labels = ssd_loss.encode_target(target, priors)
        self.assertEqual(list(target_labels[0]), [0, 0, 0, 0, 0])
        self.assertEqual(list(target_labels[1]), [0, 0, 0, 0, 0])

    def test_no_target_in_batch(self):
        prior_box = MagicMock(return_value=torch.tensor([[0, 0, 1, 1]], dtype=torch.float))
        ssd_loss = SSDLoss(5, prior_box)

        predictions = [(torch.zeros(1, 4, 1, 1), torch.zeros(1, 6, 1, 1))]
        target = [[]]
        loss = ssd_loss.forward(predictions, target)
        self.assertEqual(loss, 0)

    def test_no_target_in_one_example(self):
        prior_box = MagicMock(return_value=torch.tensor([[0, 0, 1, 1]], dtype=torch.float))
        ssd_loss = SSDLoss(5, prior_box)

        predictions = [(torch.zeros(2, 4, 1, 1), torch.zeros(2, 6, 1, 1))]
        target = [[], [[0, 0, 0, 1, 1]]]
        loss = ssd_loss.forward(predictions, target)
        self.assertNotEqual(loss, 0)


class TestSSDSigmoidLoss(unittest.TestCase):
    def test_hard_negative_mining(self):
        def gen_prior_box(x):
            return [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]  # Dummy

        ssd_loss = SSDSigmoidLoss(3, gen_prior_box)

        pred_classification = torch.tensor([[[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]]])
        target_classification = torch.tensor([[0, 0, 1]])
        mask = ssd_loss.hard_negative_mining(pred_classification, target_classification, neg_pos_ratio=0)
        self.assertEqual(mask.tolist(), [[False, False, True]])

        mask = ssd_loss.hard_negative_mining(pred_classification, target_classification, neg_pos_ratio=1)
        self.assertEqual(mask.tolist(), [[False, True, True]])

        mask = ssd_loss.hard_negative_mining(pred_classification, target_classification, neg_pos_ratio=2)
        self.assertEqual(mask.tolist(), [[True, True, True]])

    def test_one_hot(self):
        target = torch.tensor([0, 1, 2, 3])
        result = SSDSigmoidLoss._get_one_hot(target, 3, torch.float, target.layout, target.device)
        self.assertTrue(torch.all(result == torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])))

        target = torch.tensor([0])
        result = SSDSigmoidLoss._get_one_hot(target, 3, torch.float, target.layout, target.device)
        self.assertTrue(torch.all(result == torch.tensor([[0, 0, 0]])))

        target = torch.tensor([0, 0])
        result = SSDSigmoidLoss._get_one_hot(target, 3, torch.float, target.layout, target.device)
        self.assertTrue(torch.all(result == torch.tensor([[0, 0, 0], [0, 0, 0]])))


if __name__ == '__main__':
    unittest.main()
