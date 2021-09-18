import unittest
import torch
from mitorch.models.modules import YoloLoss


class TestYoloLoss(unittest.TestCase):
    def test_encode_target(self):
        yolo_loss = YoloLoss(5, None)
        target = [torch.tensor([[2, 0, 0, 1, 0.4], [3, 0, 0, 1, 1]], dtype=torch.float), torch.tensor([[4, 0, 0, 1, 0.1]], dtype=torch.float)]
        priors = torch.tensor([[0, 0, 1, 0.1],
                               [0, 0, 1, 0.4],
                               [0, 0, 1, 0.45],
                               [0, 0, 1, 0.5],
                               [0, 0, 1, 1.0]], dtype=torch.float)
        predictions = torch.zeros(2, 5, 10)
        encoded = yolo_loss.encode_target(predictions, target, priors)
        self.assertEqual(encoded[0, 1].tolist(), [0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        self.assertEqual(encoded[0, 4].tolist(), [0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(encoded[1, 0].tolist(), [0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

    def test_encode_target_with_no_target(self):
        yolo_loss = YoloLoss(5, None)
        target = [torch.tensor([[]], dtype=torch.float).reshape(0, 5), torch.tensor([[]], dtype=torch.float).reshape(0, 5)]
        priors = torch.tensor([[0, 0, 1, 0.1],
                               [0, 0, 1, 0.4],
                               [0, 0, 1, 0.45],
                               [0, 0, 1, 0.5],
                               [0, 0, 1, 1.0]], dtype=torch.float)
        predictions = torch.randn(2, 50, 1, 1)
        encoded = yolo_loss.encode_target(predictions, target, priors)
        self.assertEqual(torch.count_nonzero(encoded), 0)


if __name__ == '__main__':
    unittest.main()
