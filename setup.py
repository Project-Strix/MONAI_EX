from setuptools import setup, find_namespace_packages

setup(name='monai_ex',
      packages=find_namespace_packages(include=["monai_ex", "monai_ex.*"]),
      version='0.0.5',
      description='MONAI extension',
      author='Chenglong Wang',
      author_email='clwang@phy.ecnu.edu.cn',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "h5py",
            "matplotlib==3.3.3",
            "torch>=1.6.0",
            "monai==0.8.0",
      ],
      keywords=['deep learning', 'monai', 'pytorch']
      )