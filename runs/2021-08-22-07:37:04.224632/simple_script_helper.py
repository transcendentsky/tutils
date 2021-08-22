"""
    This script is for logging and saving details of simple scripts
    goals:
        - save the running script itself!
        - save the logging
"""

from tutils import trans_args, trans_init, load_yaml, dump_yaml, trans_configure
from tutils.framework import CSVLogger
import os
import shutil
from collections import OrderedDict


def _clear_config(config):
    if type(config) is dict or type(config) is OrderedDict:
        pop_key_list = []
        for key, value in config.items():
            print("debug: ", key, value)
            if value is None or value == "" or value == "None":
                print("debug: poped", key, value)
                pop_key_list.append(key)
            elif type(value) is dict or type(config) is OrderedDict:
                _clear_config(value)
            else:
                pass
        for key in pop_key_list:
            config.pop(key)
    return config


if __name__ == '__main__':
    # args = trans_args()
    # logger, config = trans_init(args)
    logger, config = trans_configure()
    runs_dir = config['runs_dir']
    file_path = os.path.abspath(__file__)
    shutil.copy(file_path, os.path.join(runs_dir, __file__))

    print(file_path)
    print(__file__)

    conf = load_yaml("tmp.yml")
    print(conf)
    cc = _clear_config(conf)
    print(cc)