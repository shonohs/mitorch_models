import setuptools

setuptools.setup(name='shtorch-models',
                 version='0.0.0',
                 description='Collection of pytorch models focused on readability',
                 packages=setuptools.find_namespace_packages(include=['shtorch.models', 'shtorch.models.*']),
                 license='MIT')
