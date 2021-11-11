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
# from pathlib import Path
# import argparse
import sys
from collections import OrderedDict
from .functools import _get_time_str
from .csv_recorder import CSVLogger

INFO = INFO
DEBUG = DEBUG
WARNING = WARNING
CRITICAL = CRITICAL
ERROR = ERROR


class MultiLogger(Logger):
    def __init__(self, logdir, mode=None, tag="MyLogger",extag=None, level=logging.INFO, action='k', file_name='log.log'):
        """
        mode: "wandb", "tb" or "tensorboard", "csv" : ["wandb", "tensorboard", "csv"]
        """
        super(MultiLogger, self).__init__(tag)
        self.logdir = logdir
        self.mode = "logging_only" if mode is None else mode
        self.multi_recorder = MultiRecorder(mode, logdir, tag)
        
        # --------- Standard init        
        self.propagate = False
        self.setLevel(level)
        handler = logging.StreamHandler()
        # handler = logging.FileHandler('test.log', 'w', 'utf-8') # or whatever
        handler.setFormatter(_MyFormatter(tag=tag, extag=extag, datefmt='%Y-%m-%d %H:%M:%S'))
        self.addHandler(handler)
        set_logger_dir(self, logdir, action, file_name, tag=tag, extag=extag)
    
    def add_scalars(self, *args, **kwargs):
        self.multi_recorder.add_scalars(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.info(*args, **kwargs)


class MultiRecorder(object):
    """
        Extra Log Writer , including TensorBoard, wandb
    """
    def __init__(self, mode, logdir, tag="Log") -> None:
        super().__init__()

        self.step = -1
        self.wandb_logger = None
        self.tb_logger = None
        self.csv_logger = None
        self.text_logger = None

        if mode == None: mode = []
        if type(mode) is str: mode = [mode]
        if "wandb" in mode:
            import wandb
            wandb.init(project=tag)
            wandb.watch_called = False
            self.wandb_logger = wandb
        if "tb" in mode or "tensorboard" in mode:
            from tensorboardX import SummaryWriter
            if logdir is None:
                print(f"Failed to turn on Tensorboard due to logdir=None")
            else:
                print(f"Use Tensorboard, log at '{os.path.join(logdir, 'tb')}'")
                self.tb_logger = SummaryWriter(logdir=os.path.join(logdir, "tb"))
        # if "csv" in mode:
        #     self.csv_logger = CSVLogger(logdir=os.path.join(logdir, "csv"))
        self.extra_logger = True if len(mode) > 0 else False
        
    def add_scalars(self, dicts:dict={}, step=-1, tag="train", verbose=True):
        if not self.extra_logger:
            return 
        if self.wandb_logger is not None:
            self.wandb_logger.log(dicts)

        if self.tb_logger is not None:
            if step < 0:
                step = self.step
            self.tb_logger.add_scalars(tag, dicts, global_step=step)
            
        self.step = self.step + 1

        if self.text_logger is not None:
            if verbose:
                string = f"[tlog] Step:{self.step}  "
                for key, value in dicts.items():
                    string += f"{key}:{value};"
                self.info(string)


class _MyFormatter(logging.Formatter):
    def __init__(self, tag=None, extag=None, colorful=True, *args, **kwargs):
        self.tag = tag
        self.extag = extag
        self.colorful = colorful
        extag = '-' + extag if (extag is not None and extag != '') else ''
        # print(tag, extag)
        self.taginfo = self._colored_str(f'[{tag}{extag}]', 'cyan') if tag is not None else ''
        super(_MyFormatter, self).__init__(*args, **kwargs)
        
    def format(self, record):    
        if not self.colorful:
            # Logging file
            date = self._colored_str('[%(asctime)s @%(filename)s:%(lineno)d] ', 'green')
        else:
            # Terminal
            date = self._colored_str('[%(asctime)s] ', 'green')
        date = date + self.taginfo
        msg = '%(message)s'
        if record.levelno == logging.WARNING or record.levelno == logging.DEBUG:
            fmt = date + ' ' + self._colored_str('WRN', 'yellow', attrs=['blink']) + '' + msg + ' \nPath: [%(pathname)s] ' + \
                 '\nProcess: [%(process)d %(processName)s]' + '\nThread: [%(thread)d %(threadName)s]'
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + self._colored_str('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg + ' \nPath: [%(pathname)s] ' + \
                 '\nProcess: [%(process)d %(processName)s]' + '\nThread: [%(thread)d %(threadName)s]'
        # elif record.levelno == logging.INFO:
        #     fmt = date + ' ' + self._colored_str('INFO', 'cyan', attrs=['bold']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

    def _colored_str(self, text, *args, **kwargs):
        if self.colorful:
            return colored(text, *args, **kwargs)
        else:
            return text


def _set_file(logger, path, tag=None, extag=None):
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        logger.info("Existing log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(tag=tag, extag=extag, datefmt='%Y-%m-%d-%H:%M:%S', colorful=False))

    _FILE_HANDLER = hdl
    logger.addHandler(hdl)
    logger.info("Argv: python " + ' '.join(sys.argv))


def set_logger_dir(logger, dirname='log', action='k', file_name='log.log', tag=None, extag=None):
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
    _mkdir_p(dirname)
    _set_file(logger, os.path.join(dirname, file_name), tag=tag, extag=extag)


def _mkdir_p(dirname):
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
