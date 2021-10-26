"""
    usage:
        args = trans_args()
        logger, config = trans_init(args)
    or:
        args = trans_args(parser)
        logger, config = trans_init(args, ex_config=BASE_CONFIG)
    

"""
# from termcolor import colored
import yaml
import yamlloader
# from pathlib import Path
import argparse
import sys
from collections import OrderedDict
from datetime import datetime
import os

from .tlogger import MultiLogger
from .tutils import dump_yaml
from .functools import _clear_config, _print_dict
import shutil

BASE_CONFIG = {
    'base': {
        'base_dir'  : './runs_debug/',
        'experiment': '',
        'tag': '',
        'stage': '', 
        'extag': '',
        },    
    'logger':{
        'mode': None, # "wandb", "tensorboard", 'csv'
        'action': 'k', 
    }
}


def trans_configure(config=BASE_CONFIG, **kwargs):
    # -------------  Initialize  -----------------
    config = _check_config(config)
    # if verbose:
    #     print("------  Config  ------")
    #     _print_dict(config['base'])
    # Create Logger
    logger = MultiLogger(logdir=config['base']['runs_dir'], mode=config['logger']['mode'], flag=config['base']['tag'], extag=config['base']['extag'], action=config['logger']['action']) # backup config.yaml
    config['base']['__INFO__']['logger'] = logger.mode
    config['base']['__INFO__']['Argv'] = "Argv: python " + ' '.join(sys.argv)
    dump_yaml(logger, _clear_config(config), path=config['base']['runs_dir'] + "/config.yaml")
    return logger, config


def _check_config(config):    
    config_base = config['base']
    config['base']['tag'] = config['base']['tag'] if ('tag' in config['base'].keys()) and (config['base']['tag']!="") else str(datetime.now()).replace(' ', '-')
    config['base']['extag'] = config['base']['extag'] if 'extag' in config['base'].keys() else None
    config['base']['__INFO__'] = {}
    config['base']['__INFO__']['runtime'] = str(datetime.now()).replace(' ', '-')

    experiment = config['base']['experiment'] if 'experiment' in config['base'].keys() else ''
    stage = config['base']['stage'] if 'stage' in config['base'].keys() else ''

    config['base']['runs_dir'] = os.path.join(config['base']['base_dir'], experiment, config['base']['tag'], stage)
    if os.path.exists(config['base']['runs_dir']):
        backup_name = config['base']['runs_dir'] + '.' + _get_time_str()
        shutil.move(config['base']['runs_dir'], backup_name)
    else:
        print(f"Make dir '{config['base']['runs_dir']}' !")
        os.makedirs(config['base']['runs_dir'])

    return config


def trans_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Unwarp Film Train Configure')
    try:
        parser.add_argument("-t", "--tag", type=str, default="")
    except:
        print("Already add '--tag' ")
    try:
        parser.add_argument("-et", "--extag", type=str, default="")
    except:
        print("Already add '--extag' ")
    try:
        parser.add_argument("-c", "--config", type=str, default='./configs/config.yaml') 
    except:
        print("Already add '--config' ")
    try: 
        parser.add_argument("--exp", type=str, default='', help="experiment name")
    except:
        print("Already add '--exp' ")
    try:
        parser.add_argument("-st", "--stage", type=str, default="", help="stage name for multi-stage experiment ")
    except:
        print("Already add '--stage' ")
    try:
        parser.add_argument("--test", action="store_true")
    except:
        print("Already add '--test' ")
    try:
        parser.add_argument("--func", type=str, default="", help=" function name for test specific funciton ")
    except:
        print("Already add '--func' ")
    
    args = parser.parse_args()
    return args   


def trans_init(args=None, ex_config=None, clear_none=True, **kwargs):
    """
    logger, config, tag, runs_dir = trans_init(args)
    mode: "wandb", "tb" or "tensorboard", ["wandb", "tensorboard"]
    
    action: "d": delete the directory. Note that the deletion may fail when
                the directory is used by tensorboard.
                "k": keep the directory. This is useful when you resume from a
                previous training and want the directory to look as if the
                training was not interrupted.
                Note that this option does not load old models or any other
                old states for you. It simply does nothing.
                "b" : copy the old dir
                "n" : New an new dir by time
    """
    # Load yaml config file
    config=BASE_CONFIG
    #  --------  args.config < args < ex_config  ----------
    if args is not None: 
        with open(args.config) as f:
            file_config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    else:
        file_config = {}
    ex_config = ex_config if ex_config is not None else {}
    # Clear some vars with None or ""
    arg_dict = {'base': vars(args)}
    if clear_none:
        arg_dict    = _clear_config(arg_dict)
        ex_config   = _clear_config(ex_config)

    config = merge_cascade_dict([config, file_config, arg_dict, ex_config])
    return trans_configure(config, **kwargs)


def merge_cascade_dict(dicts):
    num_dict = len(dicts)
    ret_dict = {}
    for d in dicts:
        ret_dict = _merge_two_dict(ret_dict, d)
        # print("debug: ")
        # _print_dict(ret_dict['base'])
    return ret_dict

def _merge_two_dict(d1, d2):
    # Use d2 to overlap d1
    ret_dict = {**d2, **d1}
    if isinstance(d2, dict):
        for key, value in d2.items():
            if isinstance(value, dict):
                ret_dict[key] = _merge_two_dict(ret_dict[key], d2[key])
            else:
                ret_dict[key] = d2[key]
    return ret_dict 