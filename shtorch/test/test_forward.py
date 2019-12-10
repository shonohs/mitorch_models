import unittest
import torch
import numpy as np
from shtorch.models import *

class TestForward(unittest.TestCase):
    def test_efficientnetb0(self):
        self._test_model(EfficientNetB0(), EfficientNetB0.INPUT_SIZE)

    def test_efficientnetb1(self):
        self._test_model(EfficientNetB1(), EfficientNetB1.INPUT_SIZE)

    def test_efficientnetb2(self):
        self._test_model(EfficientNetB2(), EfficientNetB2.INPUT_SIZE)

    def test_efficientnetb3(self):
        self._test_model(EfficientNetB3(), EfficientNetB3.INPUT_SIZE)

    def test_efficientnetb4(self):
        self._test_model(EfficientNetB4(), EfficientNetB4.INPUT_SIZE)

    def test_efficientnetb5(self):
        self._test_model(EfficientNetB5(), EfficientNetB5.INPUT_SIZE)

    def test_efficientnetb6(self):
        self._test_model(EfficientNetB6(), EfficientNetB6.INPUT_SIZE)

    def test_efficientnetb7(self):
        self._test_model(EfficientNetB7(), EfficientNetB7.INPUT_SIZE)

    def test_mobilenetv2(self):
        self._test_model(MobileNetV2(), 224)

    def test_mobilenetv3(self):
        self._test_model(MobileNetV3(), 224)

    def test_mobilenetv3small(self):
        self._test_model(MobileNetV3Small(), 224)

    def test_shufflenet(self):
        self._test_model(ShuffleNet(), 224)

    def test_shufflenetv2(self):
        self._test_model(ShuffleNetV2(), 224)

    def test_squeezenet(self):
        self._test_model(SqueezeNet(), 227)

    def test_vgg_a(self):
        self._test_model(VGG_A(), VGG_A.INPUT_SIZE)

    def test_vgg_b(self):
        self._test_model(VGG_B(), VGG_B.INPUT_SIZE)

    def test_vgg_c(self):
        self._test_model(VGG_C(), VGG_C.INPUT_SIZE)

    def test_vgg_d(self):
        self._test_model(VGG_D(), VGG_D.INPUT_SIZE)

    def test_vgg_e(self):
        self._test_model(VGG_E(), VGG_E.INPUT_SIZE)

    def test_vgg16(self):
        self._test_model(VGG16(), VGG16.INPUT_SIZE)

    def _test_model(self, model, input_size):
        model.eval()
        inputs = torch.tensor(np.random.rand(1, 3, input_size, input_size), dtype=torch.float32)
        outputs = model(inputs)
        self.assertIsNotNone(outputs)

if __name__ == '__main__':
    unittest.main()
