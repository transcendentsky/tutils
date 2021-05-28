import os
from tutils import tdir, tfilename, tfuncname
import shutil


# GIT_IGNORE = "__pycache__
# */runs/*
# *runs-ana/*
# */log.log*
# */visuals/*
# *.pyc
# *.jpg
# *.png
# *.npy
# *.npz
# "

def init_dirs(config):
    for dirname in config['basic_dirs']:
        tdir(dirname)
        # Make __init__.py for each dir
        make_file(tfilename(dirname, '__init__.py'))

def make_file(name):
    with open(name, 'w') as f:
        pass

def build_project():
    CONFIG = {
    "basic_dirs": [
        'configs', 'datasets', 'networks', 'utils', 'scripts', 'tmp', 'runs'
        ]
    }
    current_dir = os.path.dirname(__file__)
    shutil.copy(current_dir+"/config.yaml", "./configs/config.yaml")
    shutil.copy(current_dir+"/.gitignore", "./.gitignore")
    # Init basic dirs
    init_dirs(CONFIG)
    # TODO: create .gitignore for Git
    pass


if __name__ == '__main__':
    build_project()
