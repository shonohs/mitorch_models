import unittest
from mitorch.models import ModelFactory


class TestModelFactory(unittest.TestCase):
    def test_all(self):
        for m in ModelFactory.PREDEFINED_MODELS.keys():
            model = ModelFactory.create(m, 1)
            self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
