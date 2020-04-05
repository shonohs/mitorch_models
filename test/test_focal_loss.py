import unittest
import torch
from mitorch.models.modules import FocalLoss


class TestFocalLoss(unittest.TestCase):
    def test_all_background(self):
        loss = FocalLoss(5, prior_box=None)

        # both predicted and target is all background classes
        pred_classification = torch.zeros((1, 10, 5))
        target_classification = torch.zeros((1, 10), dtype=torch.long)
        pred_classification[:] = -1000

        result = loss.loss_classification(pred_classification, target_classification)
        self.assertEqual(result, 0)

        # batch_size=8
        pred_classification = torch.zeros((8, 100, 5))
        target_classification = torch.zeros((8, 100), dtype=torch.long)
        pred_classification[:] = -1000

        result = loss.loss_classification(pred_classification, target_classification)
        self.assertEqual(result, 0)

        # num_classes = 1
        pred_classification = torch.zeros((1, 10, 1))
        target_classification = torch.zeros((1, 10), dtype=torch.long)
        pred_classification[:] = -1000

        loss = FocalLoss(1, prior_box=None)
        result = loss.loss_classification(pred_classification, target_classification)
        self.assertEqual(result, 0)

    def test_all_true(self):
        loss = FocalLoss(5, prior_box=None)

        # both predicted and target is all background classes
        pred_classification = torch.zeros((1, 10, 5))
        target_classification = torch.ones((1, 10), dtype=torch.long)
        pred_classification[:, :, 1:] = -1000
        pred_classification[:, :, 0] = 1000

        result = loss.loss_classification(pred_classification, target_classification)
        self.assertEqual(result, 0)

        # batch_size=8
        pred_classification = torch.zeros((8, 100, 5))
        target_classification = torch.ones((8, 100), dtype=torch.long)
        pred_classification[:, :, 1:] = -1000
        pred_classification[:, :, 0] = 1000

        result = loss.loss_classification(pred_classification, target_classification)
        self.assertEqual(result, 0)

    def test_ignore_all(self):
        loss = FocalLoss(5, prior_box=None)
        pred_classification = torch.zeros((5, 10, 5))
        target_classification = torch.zeros((5, 10), dtype=torch.long)
        target_classification[:] = -1

        result = loss.loss_classification(pred_classification, target_classification)
        self.assertEqual(result, 0)

    def test_one_hot(self):
        target = torch.tensor([0, 1, 2, 3])
        result = FocalLoss._get_one_hot(target, 3, torch.float, target.layout, target.device)
        self.assertTrue(torch.all(result == torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])))

        target = torch.tensor([0])
        result = FocalLoss._get_one_hot(target, 3, torch.float, target.layout, target.device)
        self.assertTrue(torch.all(result == torch.tensor([[0, 0, 0]])))

        target = torch.tensor([0, 0])
        result = FocalLoss._get_one_hot(target, 3, torch.float, target.layout, target.device)
        self.assertTrue(torch.all(result == torch.tensor([[0, 0, 0], [0, 0, 0]])))


if __name__ == '__main__':
    unittest.main()
