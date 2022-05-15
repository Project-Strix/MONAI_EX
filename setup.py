from setuptools import setup, find_namespace_packages
import versioneer

setup(name='monai_ex',
      packages=find_namespace_packages(include=["monai_ex", "monai_ex.*"]),
      version=versioneer.get_version(),
      description='MONAI extension',
      author='Chenglong Wang',
      author_email='clwang@phy.ecnu.edu.cn',
      license='Apache License Version 2.0, January 2004',
      cmdclass=versioneer.get_cmdclass(),
      keywords=['deep learning', 'monai', 'pytorch']
      )