import unittest
import torch
from mitorch.models import *


class TestBidirectionalFeaturePyramidNetwork(unittest.TestCase):
    def test_basic_block_various_size(self):
        self._test_bb_forward([2, 4, 8, 16, 32])
        self._test_bb_forward([4, 8, 16, 32, 64])
        self._test_bb_forward([3, 6, 12, 25, 50])
        self._test_bb_forward([3, 5, 10, 20, 40])

    def _test_bb_forward(self, input_sizes):
        model = BidirectionalFeaturePyramidNetwork.BasicBlock([2, 2, 2, 2, 2], 2)

        inputs = [torch.zeros([1, 2, x, x]) for x in input_sizes]

        self.assertIsNotNone(model.forward(inputs))


if __name__ == '__main__':
    unittest.main()
