
import yaml
import yamlloader
import shutil
import os
from .functools import d, _get_time_str, _ordereddict_to_dict
from pathlib import Path


def save_script(runs_dir, _file_name, logger=None):
    file_path = os.path.abspath(_file_name)
    parent, name = os.path.split(_file_name)
    time = _get_time_str()
    output_path = os.path.join(runs_dir, name)

    if os.path.isfile(output_path):
        backup_name = output_path + '.' + _get_time_str()
        shutil.move(output_path, backup_name)
        print(f"Existing yaml file '{output_path}' backuped to '{backup_name}' ")

    shutil.copy(file_path, output_path)
    # with open(os.path.join(runs_dir, "save_script.log"), "a+") as f:
    #     f.write(f"Script location: {_file_name};  Time: {time}\n")
    print(f"Saved script file: from {file_path} to {output_path}")


def dump_yaml(logger, config, path=None, verbose=True):
    # Backup existing yaml file
    path = config['base']['runs_dir'] + "/config.yaml" if path is None else path
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        logger.info(f"Existing yaml file '{path}' backuped to '{backup_name}' ")
    with open(path, "w") as f:
        config = _ordereddict_to_dict(config)
        yaml.dump(config, f)
    if verbose:
        logger.info(f"Saved config.yaml to {path}")


def load_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    return config


def tdir(*dir_paths):
    def checkslash(name):
        if name.startswith("/"):
            name = name[1:]
            return checkslash(name)
        else:
            return name
    if len(dir_paths) <= 1:
        dir_path = dir_paths[0]
    else:
        names = [dir_paths[0]]
        for name in dir_paths[1:]:
            names.append(checkslash(name))
        dir_path = os.path.join(*names)
    d(dir_path)
    if not os.path.exists(dir_path):
        d("Create Dir Path: ", dir_path)
        os.makedirs(dir_path)
    if not dir_path.endswith("/"):
        dir_path += "/"
    return dir_path


def tfilename(*filenames):
    def checkslash(name):
        if name.startswith("/"):
            name = name[1:]
            return checkslash(name)
        else:
            return name
    if len(filenames) <= 1:
        filename = filenames[0]
    else:
        names = [filenames[0]]
        for name in filenames[1:]:
            names.append(checkslash(name))
        filename = os.path.join(*names)
    d(filename)
    parent, name = os.path.split(filename)
    if parent != '' and not os.path.exists(parent):
        d(parent)
        os.makedirs(parent)
    return filename

def tfilename2(*filenames):
    path = Path(os.path.join(*filenames))
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path.cwd() / path

def tdir2(*dir_paths):
    path = Path(os.path.join(*dir_paths))
    path.mkdir(parents=True, exist_ok=True)
    return Path.cwd() / path

def texists(*filenames):
    path = os.path.join(*filenames)
    return os.path.exists(path)


def add_total(tuple1, tuple2):
    l = list()
    for i, item in enumerate(tuple1):
        l.append(tuple1[i] + tuple2[i])
    return tuple(l)


if __name__ == "__main__":
    d(adadada="tutils")
