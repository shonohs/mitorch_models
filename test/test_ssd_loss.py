import unittest
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


if __name__ == '__main__':
    unittest.main()
