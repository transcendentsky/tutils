from setuptools import setup, find_packages

setup(
    name="trans-utils",
    version="0.2.1",
    author="Curliq",
    author_email="transcendentsiki@gmail.com",
    packages=find_packages(exclude=['test.*']),
    package_data={'tutils': ['*/config.yaml', '*/.gitignore', '*/NOTES.md', '*/ttest.csv'],
                  'proj-template': ['*/code/configs/ablation.yaml', '*/code/configs/config.yaml', '*/code/.gitignore']},
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
      'h5py',
      'xlwt',
      'openpyxl',
      # 'piq', # for ssim_loss
      # 'pyradiomics', # for radiomics
      # 'kornia>=0.4.0', 
  ],
  entry_points={
    'console_scripts': [
      'trans_build = tutils.tutils.build_proj:main',
      'abla_train = tutils.trainer.ablation_exec:execute'
    ],
  },
)
# py_modules=['tutils'],