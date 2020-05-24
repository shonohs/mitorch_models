import unittest
import torch
from mitorch.models.modules import PriorBox


class TestPriorBox(unittest.TestCase):
    def test_one_scale(self):
        prior_box = PriorBox(1, aspect_ratios=[2])
        dummy_input = [(torch.zeros(1, 100, 20, 20), torch.zeros(1, 50, 20, 20))]

        boxes = prior_box(dummy_input)

        self.assertEqual(prior_box.get_num_priors(), [4])
        self.assertEqual(list(boxes.shape), [1600, 4])
        self.assertFalse(torch.isnan(boxes).any())
        self.assertFalse(torch.isinf(boxes).any())

    def test_size1_input(self):
        prior_box = PriorBox(1, aspect_ratios=[2])
        dummy_input = [(torch.zeros(1, 100, 1, 1), torch.zeros(1, 50, 1, 1))]

        boxes = prior_box(dummy_input)

        self.assertEqual(prior_box.get_num_priors(), [4])
        self.assertEqual(list(boxes.shape), [4, 4])
        self.assertFalse(torch.isnan(boxes).any())
        self.assertFalse(torch.isinf(boxes).any())

        centers = (boxes[:, 0:2] + boxes[:, 2:]) / 2
        self.assertEqual(centers.tolist(), [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    def test_wrong_scale_order(self):
        prior_box = PriorBox(1, aspect_ratios=[2])
        dummy_input = [(torch.zeros(1, 100, 20, 20), torch.zeros(1, 50, 20, 20)),
                       (torch.zeros(1, 100, 40, 40), torch.zeros(1, 50, 40, 40))]

        with self.assertRaises(Exception):
            prior_box(dummy_input)


if __name__ == '__main__':
    unittest.main()
