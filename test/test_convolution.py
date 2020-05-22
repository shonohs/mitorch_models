import unittest
import torch
from mitorch.models.modules import Conv2dAct, default_module_settings


class TestConvolution(unittest.TestCase):
    def test_none_activation_priority(self):
        @default_module_settings(**{'!activation': 'relu'})
        def make_convolution():
            return Conv2dAct(3, 3, 3, activation='none')

        conv = make_convolution()
        self.assertIsNone(conv.activation)

        @default_module_settings(**{'!activation': 'relu'})
        def make_convolution2():
            return Conv2dAct(3, 3, 3, activation='relu6')

        conv = make_convolution2()
        self.assertIsInstance(conv.activation, torch.nn.ReLU)


if __name__ == '__main__':
    unittest.main()
