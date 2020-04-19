import unittest
from mitorch.models.modules import ModuleBase, default_module_settings


class TestModule(unittest.TestCase):
    def test_empty_settings(self):
        module = ModuleBase()
        self.assertIsInstance(module.module_settings, dict)

    def test_kwargs(self):
        module = ModuleBase(testparam=1, testparam2=2)
        self.assertEqual(module.module_settings['testparam'], 1)
        self.assertEqual(module.module_settings['testparam2'], 2)

    def test_default_settings(self):
        @default_module_settings(testparam=1)
        def make_module():
            return ModuleBase()

        module = make_module()
        self.assertEqual(module.module_settings['testparam'], 1)

    def test_settings_priority(self):
        @default_module_settings(testparam=1)
        def make_module():
            return ModuleBase(testparam=2)

        module = make_module()
        self.assertEqual(module.module_settings['testparam'], 2)

    def test_important_settings(self):
        @default_module_settings(**{'!testparam': 1})
        def make_module():
            return ModuleBase(testparam=2)

        module = make_module()
        self.assertEqual(module.module_settings['testparam'], 1)


if __name__ == '__main__':
    unittest.main()
