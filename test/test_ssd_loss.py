import unittest
from unittest.mock import MagicMock
import torch
from mitorch.models.modules import SSDLoss


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


if __name__ == '__main__':
    unittest.main()
