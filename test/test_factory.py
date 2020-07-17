import unittest
import torch
from mitorch.models import ModelFactory


class TestModelFactory(unittest.TestCase):
    def test_all(self):
        for m in ModelFactory.PREDEFINED_MODELS.keys():
            model = ModelFactory.create(m, 1)
            self.assertIsNotNone(model)

    def test_model_options(self):
        model = ModelFactory.create('ShuffleNetV2', 3, ['relu6'])
        self.assertTrue(any(isinstance(m, torch.nn.ReLU6) for m in model.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.ReLU) for m in model.modules()))

        model = ModelFactory.create('ShuffleNetV2', 3, ['relu'])
        self.assertTrue(any(isinstance(m, torch.nn.ReLU) for m in model.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.ReLU6) for m in model.modules()))

        model = ModelFactory.create('ShuffleNetV2', 3, ['relu6', 'sync_bn'])
        self.assertTrue(any(isinstance(m, torch.nn.ReLU6) for m in model.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.ReLU) for m in model.modules()))
        self.assertTrue(any(isinstance(m, torch.nn.SyncBatchNorm) for m in model.modules()))
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))


if __name__ == '__main__':
    unittest.main()
