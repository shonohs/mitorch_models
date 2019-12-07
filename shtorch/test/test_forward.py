import unittest
import torch
import numpy as np
from shtorch.models import MobileNetV2

class TestForward(unittest.TestCase):
    def test_mobilenetv2(self):
        self._test_model(MobileNetV2(), 224)


    def _test_model(self, model, input_size):
        model.eval()
        inputs = torch.tensor(np.random.rand(1, 3, input_size, input_size), dtype=torch.float32)
        outputs = model(inputs)
        self.assertIsNotNone(outputs)

if __name__ == '__main__':
    unittest.main()
