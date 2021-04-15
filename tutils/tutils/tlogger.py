# -*- coding: utf-8 -*-
# File: logger.py

'''
    How To Use :
    from mylogger import get_mylogger, set_logger_dir, auto_set_dir
    logger = get_mylogger()
    logger.warning("warning")
    set_logger_dir(logger, "test")
'''
import errno
import logging
from logging import Logger
import os
import os.path
import shutil
import sys
from logging import INFO, DEBUG, WARNING, CRITICAL, ERROR
from datetime import datetime
from six.moves import input
from termcolor import colored
import yaml
import yamlloader
from pathlib import Path
import argparse
import sys
# import traceback
# from typing import List, Dict

INFO = INFO
DEBUG = DEBUG
WARNING = WARNING
CRITICAL = CRITICAL
ERROR = ERROR

def trans_init(args=None, mode=None):
    """
    logger, config, tag, runs_dir = trans_init(args, mode=None)
    mode: "wandb", "tb" or "tensorboard", ["wandb", "tensorboard"]
    """
    # Load yaml config file
    if args == None: 
        config=dict({'runs_dir':'./runs/',
                     'dataset_pth' : '../../dataset/Cephalometric/',
                     'batch_size' : 16,
                     'num_workers' : 8,
                     'learning_rate' : 0.0003,
                     'decay_step' : 150,
                     'decay_gamma' : 0.1,
                     'num_epochs' : 500})
        with open('./config.yaml', "w") as f:
            yaml.dump(config, f)
    else:
        try:
            with open(args.config) as f:
                config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
        except AttributeError as e:
            print(sys.exc_info())
            config = dict({'runs_dir':'./runs/'})
        except Exception as e:
            print(sys.exc_info())
            config = dict({'runs_dir':'./runs/'})
            
    # Create runs dir
    tag = str(datetime.now()).replace(' ', '-') if (args == None) or (args.tag == '') else args.tag
    extag = None if (args == None) or (args.extag == '') else args.extag
    runs_dir = config['runs_dir'] + tag
    config['runs_dir'] = runs_dir
    config['tag'] = tag
    if not os.path.exists(runs_dir):
        print(f"Make dir '{runs_dir}' !")
        os.makedirs(runs_dir)
    # Create Logger
    # logger = get_mylogger(multi=multi, flag=tag, log_dir=runs_dir)
    logger = MultiLogger(log_dir=runs_dir, mode=mode, flag=tag, extag=extag)
    logger.info(config)
    config['logger'] = logger.mode

    return logger, config

def trans_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='Unwarp Film Train Configure')
    parser.add_argument("-t", "--tag", type=str, default="")
    parser.add_argument("-et", "--extag", type=str, default="")
    parser.add_argument("-c", "--config", type=str, default='config.yaml') 
    args = parser.parse_args()
    return args   

def dump_yaml(logger, config, verbose=True):
    # Backup existing yaml file
    path = config['runs_dir'] + "/config.yaml"
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        logger.info(f"Existing yaml file '{path}' backuped to '{backup_name}' ")
    with open(path, "w") as f:
        yaml.dump(config, f)
    if verbose:
        logger.info(f"Saved config.yaml to {path}")
# def get_mylogger(multi=False, level=logging.INFO, flag="MyLogger", log_dir=None, action='k', file_name='log.log'):
#     logger = logging.getLogger(flag)
#     if multi:
#         logger = MultiLogger(flag=flag, mode="tb", log_dir=log_dir)
#     logger.propagate = False
#     logger.setLevel(level)
#     handler = logging.StreamHandler()
#     handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
#     logger.addHandler(handler)
#     if log_dir is not None:
#         set_logger_dir(logger, log_dir, action, file_name)
#     return logger

class MultiLogger(Logger):
    def __init__(self, log_dir, mode=None, flag="MyLogger",extag=None, level=logging.INFO, action='k', file_name='log.log'):
        """
        mode: "wandb", "tb" or "tensorboard", ["wandb", "tensorboard"]
        """
        super(MultiLogger, self).__init__(flag)
        self.wandb = None
        self.tb = None
        self.step = -1
        self.log_dir = log_dir
        self.mode = mode
        
        if mode == None: mode = []
        if type(mode) is str: mode = [mode]
        if "wandb" in mode:
            import wandb
            wandb.init(project=flag)
            wandb.watch_called = False
            self.wandb = wandb
        if "tb" in mode or "tensorboard" in mode:
            from tensorboardX import SummaryWriter
            if log_dir is None:
                self.logger.warning(f"Failed to turn on Tensorboard due to logdir=None")
            else:
                self.logger.info(f"Use Tensorboard, log at '{os.path.join(log_dir, 'tb')}'")
                writer = SummaryWriter(logdir=os.path.join(log_dir, "tb"))
                
        # --------- Standard init
        
        self.propagate = False
        self.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(_MyFormatter(tag=flag, extag=extag, datefmt='%m%d %H:%M:%S'))
        self.addHandler(handler)
        set_logger_dir(self, log_dir, action, file_name)

    def tlog(self, dicts:dict={}, step=-1, verbose=False):
        if self.wandb is not None:
            self.wandb.log(dicts)
        if self.tb is not None:
            if step < 0:
                step = self.step
            for key, value in dicts.items():
                self.tb.add_scaler(key, value, global_step=step)
        self.step = self.step + 1
        if verbose:
            string = f"[tlog] Step:{self.step}  "
            for key, value in dicts.items():
                string += f"{key}:{value};"
            self.info(string)

    def add_scalar(self, key, value, global_step=-1, verbose=False):
        if self.wandb is not None:
            self.wandb.log({key:value})
        if self.tb is not None:
            if global_step < 0:
                global_step = self.step
            self.tb.add_scalar(key, value, global_step)
        self.step = self.step + 1
        if verbose:
            self.info(f"[add_scalar] Step:{global_step}  {key}:{value}")
    
class _MyFormatter(logging.Formatter):
    def __init__(self, tag=None, extag=None, *args, **kwargs):
        self.tag = tag
        self.extag = extag
        extag = '-' + extag if (extag is not None and extag != '') else ''
        print(tag, extag)
        self.taginfo = colored(f' [{tag + extag}]', 'yellow') if tag is not None else ''
        super(_MyFormatter, self).__init__(*args, **kwargs)
        
    def format(self, record):
        tag = self.tag
        extag = self.extag
        
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        date = date + self.taginfo
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'yellow', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR :
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + ' ' + colored('DBG', 'magenta', attrs=['bold']) + ' ' + msg
        elif record.levelno == logging.INFO:
            fmt = date + ' ' + colored('INFO', 'cyan', attrs=['bold']) + ' ' + msg
        elif record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('CRITICAL', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')

def _set_file(logger, path):
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        logger.info("Existing log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))

    _FILE_HANDLER = hdl
    logger.addHandler(hdl)
    logger.info("Argv: " + ' '.join(sys.argv))


def set_logger_dir(logger, dirname='log', action='k', file_name='log.log'):
    """
    Set the directory for global logging.
    Args:
        dirname(str): log directory
        action(str): an action of ["k","d","q"] to be performed
            when the directory exists. Will ask user by default.
                "d": delete the directory. Note that the deletion may fail when
                the directory is used by tensorboard.
                "k": keep the directory. This is useful when you resume from a
                previous training and want the directory to look as if the
                training was not interrupted.
                Note that this option does not load old models or any other
                old states for you. It simply does nothing.
                "b" : copy the old dir
                "n" : New an new dir by time
    """
    def dir_nonempty(dirname):
        # If directory exists and nonempty (ignore hidden files), prompt for action
        return os.path.isdir(dirname) and len([x for x in os.listdir(dirname) if x[0] != '.'])

    if dir_nonempty(dirname):
        if action == 'b':
            backup_name = dirname + _get_time_str()
            shutil.move(dirname, backup_name)
            logger.info("Directory '{}' backuped to '{}'".format(dirname, backup_name))  # noqa: F821
        elif action == 'd':
            shutil.rmtree(dirname, ignore_errors=True)
            if dir_nonempty(dirname):
                shutil.rmtree(dirname, ignore_errors=False)
        elif action == 'n':
            dirname = dirname + _get_time_str()
            logger.info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif action == 'k':
            pass
        else:
            raise OSError("Directory {} exits!".format(dirname))
    mkdir_p(dirname)
    _set_file(logger, os.path.join(dirname, file_name))

def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e