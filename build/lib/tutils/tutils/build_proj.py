"""
    Dataset Preparation:
        Spacing
        Intensity Range
        Normalization: mean std 
"""

import os
from tutils import tdir, tfilename, tfuncname
import shutil

def init_dirs(config):
    for dirname in config['basic_dirs']:
        tdir(dirname)
        # Make __init__.py for each dir
        make_file(tfilename(dirname, '__init__.py'))

def make_file(name):
    with open(name, 'w') as f:
        pass

def build_project_v0():
    CONFIG = {
    "basic_dirs": [
        'configs', 'datasets', 'networks', 'utils', 'scripts', 'tmp', 'runs'
        ]
    }
    current_dir = os.path.dirname(__file__)
    # Init basic dirs
    init_dirs(CONFIG)
    shutil.copy(current_dir+"/config.yaml", "./configs/config.yaml")
    shutil.copy(current_dir+"/.gitignore", "./.gitignore")
    shutil.copy(current_dir+"/NOTES.md", "./NOTES.md")
    # TODO: create .gitignore for Git
    pass


def build_project_v1():
    current_dir = os.path.dirname(__file__)
    for _dir in os.scandir(current_dir + "/../proj-template/"):
        if not os.path.exists(_dir.path):
            if os.path.isdir(_dir.path):
                shutil.copytree(_dir.path, f"./{_dir.name}")
            else:
                shutil.copy(_dir.path,  f"./{_dir.name}")
        # shutil.copytree(current_dir + "/proj-template/", f"./")

    # shutil.copy(current_dir + "/data", "./data")
    # shutil.copy(current_dir + "/runs", "./runs")

def build_project_v2():
    dl_proj = "https://gitee.com/transcendentsky/dl-proj.git"
    os.system(f"git clone {dl_proj}")


def main():
    build_project_v1()
    

if __name__ == '__main__':
    main()
