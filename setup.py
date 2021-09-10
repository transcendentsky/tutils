from setuptools import setup, find_packages

setup(
    name="tutils",
    version="1.2.0",
    author="Curliq",
    author_email="transcendentsiki@gmail.com",
    packages=find_packages(exclude=['test.*']),
    package_data={'tutils': ['*/config.yaml', '*/.gitignore', '*/NOTES.md', '*/ttest.csv'],
                  'proj': ['*/proj-template/*', '*/proj-template/code/configs/config.yaml', ]},
    install_requires=[
      'termcolor',
      'einops',
      'torch>=1.6',
      'torchvision',
      'seaborn',
      'matplotlib',
      'opencv-python',
      'pandas',
      'pyyaml',
      'yamlloader',
      'SimpleITK',
      # 'piq', # for ssim_loss
      # 'pyradiomics', # for radiomics
      # 'kornia>=0.4.0', 
  ],
  entry_points={
    'console_scripts': [
      'trans_build = tutils.tutils.build_proj:main',
    ],
  },
)
# py_modules=['tutils'],