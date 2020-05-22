import unittest
import torch
from mitorch.models.modules import DepthwiseSeparableConv2d, default_module_settings


class TestConvolution(unittest.TestCase):
    def test_none_activation_priority(self):
        @default_module_settings(**{'!activation': 'relu', '!activation2': 'relu'})
        def make_convolution():
            return DepthwiseSeparableConv2d(3, 3, 3, activation='none', activation2='none')

        conv = make_convolution()
        self.assertIsNone(conv.depthwise_conv.activation)
        self.assertIsNone(conv.pointwise_conv.activation)

        @default_module_settings(**{'!activation': 'relu', '!activation2': 'relu'})
        def make_convolution2():
            return DepthwiseSeparableConv2d(3, 3, 3, activation='relu6', activation2='relu6')

        conv = make_convolution2()
        self.assertIsInstance(conv.depthwise_conv.activation, torch.nn.ReLU)
        self.assertIsInstance(conv.pointwise_conv.activation, torch.nn.ReLU)


if __name__ == '__main__':
    unittest.main()
