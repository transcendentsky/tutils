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
from .tutils import dump_yaml, save_script
from .functools import _clear_config, _print_dict
import copy



BASE_CONFIG = {
    'base': {
        'base_dir'  : './runs_debug/',
        'experiment': "test_param1",
        'tag': '',
        'stage': '', 
        'extag': '',
        'config': '',
        'test': False,
        'func': '',
        'gpus': -1,
        },    
    'logger':{
        'mode': None, # "wandb", "tensorboard", 'csv'
        'action': 'k', 
    }
}


def trans_configure(config=BASE_CONFIG, file=None, **kwargs):
    if file is not None:
        parent, name = os.path.split(file)
        name = name[:-3]
        print(f"Change experiment name from {config['base']['experiment']} to {name}")
        config['base']['experiment'] = name

    # -------------  Initialize  -----------------
    config = _check_config(config)
    # if verbose:
    #     print("------  Config  ------")
    #     _print_dict(config['base'])
    # Create Logger
    config_base = config['base']
    config_logger = config['logger']
    # _print_dict(config)
    logger = MultiLogger(logdir=config_base['runs_dir'], 
                         mode=config_logger.get('mode', None), 
                         tag=config_base['tag'], 
                         extag=config_base.get('experiment', None),
                         action=config_logger.get('action', 'k')) # backup config.yaml
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
    if not os.path.exists(config['base']['runs_dir']):
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
        parser.add_argument("-exp", "--experiment", type=str, default='', help="experiment name")
    except:
        print("Already add '--experiment' ")
    try:
        parser.add_argument("-st", "--stage", type=str, default="", help="stage name for multi-stage experiment ")
    except:
        print("Already add '--stage' ")
    try:
        parser.add_argument("--test", action="store_true")
    except:
        print("Already add '--test' ")
    try:
        parser.add_argument("--func", type=str, default="train", help=" function name for test specific funciton ")
    except:
        print("Already add '--func' ")
    try:
        parser.add_argument("--ms", action="store_true", help=" Turn on Multi stage mode ! ")
    except:
        print("Already add '--ms' ")
    try:
        parser.add_argument("--gpus", type=int, default=-1, help=" Turn on Multi stage mode ! ")
    except:
        print("Already add '--gpus' ")
    args = parser.parse_args()
    return args   


def trans_init(args=None, ex_config=None, file=None, clear_none=True, **kwargs):
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
    arg_dict_special = vars(copy.deepcopy(args))
    for k, v in config['base'].items():
        if k in arg_dict_special.keys():
            config['base'][k] = arg_dict_special.pop(k)
    arg_dict = {'special': arg_dict_special}
    print("debug: arg-dict")
    _print_dict(arg_dict)

    if clear_none:
        file_config = _clear_config(file_config)
        arg_dict    = _clear_config(arg_dict)
        ex_config   = _clear_config(ex_config)

    config = merge_cascade_dict([config, file_config, arg_dict, ex_config])

    return trans_configure(config, file=file, **kwargs)


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