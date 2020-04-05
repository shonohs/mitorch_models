import unittest
import torch
from mitorch.models import MobileNetV2
from mitorch.models.heads import MnasFPN


class TestMnasFPN(unittest.TestCase):
    def test_mobilenetv2(self):
        self._test_model(MobileNetV2, 320)

    def _test_model(self, model_class, input_size):
        model = MnasFPN(model_class())
        model.eval()
        inputs = torch.randn((1, 3, input_size, input_size), dtype=torch.float32)
        outputs = model(inputs)
        self.assertIsNotNone(outputs)


if __name__ == '__main__':
    unittest.main()
