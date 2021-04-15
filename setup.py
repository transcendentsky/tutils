from setuptools import setup, find_packages

setup(
    name="tutils",
    version="1.0.1",
    author="trans",
    author_email="transcendentsiki@gmail.com",
    packages=find_packages(),
    install_requires=[
      'termcolor',
      'einops',
      'torch>=1.6',
      'kornia>=0.4.0',
      'torchvision',
      'seaborn',
      'matplotlib',
      'opencv-python',
      'pandas',
      'pyyaml',
      'yamlloader',
      'SimpleITK',
      # 'piq',
      # 'pyradiomics',
  ],
)
# py_modules=['tutils'],