import unittest
from mitorch.models.modules import ModuleBase, default_module_settings, set_module_settings


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

    def test_set_module_settings(self):
        with set_module_settings(testparam=1):
            module = ModuleBase()
        self.assertEqual(module.module_settings['testparam'], 1)

        @default_module_settings(testparam=1, testparam2=2)
        def make_module():
            return ModuleBase()

        with set_module_settings(testparam=3):
            module = make_module()
        self.assertEqual(module.module_settings, {'testparam': 3, 'testparam2': 2})

        with set_module_settings(**{'!testparam': 5, 'testparam2': 5}):
            module = make_module()

        self.assertEqual(module.module_settings, {'testparam': 5, 'testparam2': 5})


if __name__ == '__main__':
    unittest.main()
