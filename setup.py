import setuptools

setuptools.setup(name='shtorch-models',
                 version='0.0.0',
                 description='Collection of pytorch models focused on readability',
                 url='https://github.com/shonohs/shtorch_models',
                 packages=setuptools.find_namespace_packages(include=['shtorch.models', 'shtorch.models.*']),
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3 :: Only',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 license='MIT')
