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

BASE_CONFIG = {
    'base_dir': './runs/'
}


def trans_configure(config=BASE_CONFIG, mode=None, action='k', **kwargs):
    # -------------  Initialize  -----------------
    config['tag'] = config['tag'] if ('tag' in config.keys()) and (config['tag']!="") else str(datetime.now()).replace(' ', '-')
    config['extag'] = config['extag'] if 'extag' in config.keys() else None
    config['__INFO__'] = {}
    config['__INFO__']['runtime'] = str(datetime.now()).replace(' ', '-')

    runs_dir = os.path.join(config['base_dir'], config['tag'])
    config['runs_dir'] = runs_dir
    if not os.path.exists(runs_dir):
        print(f"Make dir '{runs_dir}' !")
        os.makedirs(runs_dir)
    # Create Logger
    logger = MultiLogger(log_dir=runs_dir, mode=mode, flag=config['tag'], extag=config['extag'], action=action) # backup config.yaml
    _print_dict(config)
    config['__INFO__']['logger'] = logger.mode
    config['__INFO__']['Argv'] = "Argv: python " + ' '.join(sys.argv)
    _print_dict(config['__INFO__'])
    dump_yaml(logger, config)
    return logger, config




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
        parser.add_argument("-st", "--stage", type=str, default="")
    except:
        print("Already add '--stage' ")
    try:
        parser.add_argument("--test", action="store_true")
    except:
        print("Already add '--test' ")
    
    args = parser.parse_args()
    return args   

def trans_init(args=None, ex_config=None, mode=None, action='k', clear_none=True, **kwargs):
    """
    logger, config, tag, runs_dir = trans_init(args, mode=None)
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
    config=dict({'base_dir':'../runs_debug/',})
    #  --------  args.config < args < ex_config  ----------
    if args is not None: 
        with open(args.config) as f:
            args_config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    else:
        args_config = {}
    ex_config = ex_config if ex_config is not None else {}
    # Clear some vars with None or ""
    args = vars(args)
    if clear_none:
        args        = _clear_config(args)
        ex_config   = _clear_config(ex_config)
    # Integrate all settings
    config = {**config, **args_config, **args, **ex_config}
    return trans_configure(config, mode=mode, action='k', **kwargs)

