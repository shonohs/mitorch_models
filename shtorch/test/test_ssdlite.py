import unittest
import torch
import numpy as np
from shtorch.models import *

class TestSSDLite(unittest.TestCase):
    def test_efficientnetb0(self):
        self._test_model(EfficientNetB0, 320)

    def test_mobilenetv2(self):
        self._test_model(MobileNetV2, 320)

    def test_mobilenetv3(self):
        self._test_model(MobileNetV3, 320)

    def test_mobilenetv3small(self):
        self._test_model(MobileNetV3Small, 320)

    def test_seresnext50(self):
        self._test_model(SEResNext50, 320)

    def test_shufflenetv2(self):
        self._test_model(ShuffleNetV2, 320)

    def test_vgg16(self):
        self._test_model(VGG16ForSSD, 320)

    def _test_model(self, model_class, input_size):
        model = SSDLite(SSDLiteExtraLayers(model_class()), 3)
        model.eval()
        inputs = torch.tensor(np.random.rand(1, 3, input_size, input_size), dtype=torch.float32)
        outputs = model(inputs)
        self.assertIsNotNone(outputs)
        predictions = model.predictor(outputs)
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(model.loss)

if __name__ == '__main__':
    unittest.main()
