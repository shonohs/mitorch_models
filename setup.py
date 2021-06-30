import setuptools

setuptools.setup(name='mitorch-models',
                 version='0.0.4',
                 author='shono',
                 description='Collection of vision-related pytorch models with a focus on code readability',
                 url='https://github.com/shonohs/mitorch_models',
                 packages=setuptools.find_namespace_packages(include=['mitorch.models', 'mitorch.models.*']),
                 install_requires=['torch>=1.4'],
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3 :: Only',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence'
                 ],
                 license='MIT')
