import os
import numpy as np
# import torch
import random
import torchvision
import string

import random
import time
import cv2

from pathlib import Path
from datetime import datetime
from collections import OrderedDict



def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')


def _clear_config(config):
    # if type(config) is dict or type(config) is OrderedDict:
    if isinstance(config, (dict, OrderedDict)):
        pop_key_list = []
        for key, value in config.items():
            # print("debug: ", key, value)
            if value is None or value == "" or value == "None":
                # print("debug: poped", key, value)
                pop_key_list.append(key)
            elif isinstance(config, (dict, OrderedDict)):
                _clear_config(value)
            else:
                pass
        for key in pop_key_list:
            config.pop(key)
    return config


def print_dict(config):
    _print_dict(config)


def _print_dict(_dict, layer=0):
    if isinstance(_dict, (dict, OrderedDict)):
        for key, value in _dict.items():
            if isinstance(value, (dict, OrderedDict)):
                print("    "*layer, key, end=":\n")
                _print_dict(value, layer+1)
            else:
                print("    "*layer, f"{key}: {value}")
    else:
        print("    "*layer, _dict)


# def time_now():
#     return time.strftime("%Y%m%d-%H%M%S", time.localtime())


# def generate_random_str(n: int = 6):
#     ran_str = ''.join(random.sample(string.ascii_letters + string.digits, n))
#     return ran_str


# def generate_name():
#     return time_now() + '-' + generate_random_str(6)



def _ordereddict_to_dict(d):
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if type(v) == OrderedDict:
            v = _ordereddict_to_dict(v)
            d[k] = dict(v)
        elif type(v) == list:
            d[k] = _ordereddict_to_dict(v)
        elif type(v) == dict:
            d[k] = _ordereddict_to_dict(v)
    return d








######################################################

class Config(object):
    def __init__(self):
        super().__init__()
        self.TUTILS_DEBUG = False
        self.TUTILS_INFO = False
        self.TUTILS_WARNING = True

    def set_print_debug(self, setting=True):
        self.TUTILS_DEBUG = setting

    def set_print_info(self, setting=True):
        self.TUTILS_INFO = setting

    def set_print_warning(self, setting=True):
        self.TUTILS_WARNING = setting

tconfig = Config()

def tprint(*s, end="\n", **kargs):
    if len(s) > 0:
        for x in s:
            print(x, end="")
        print("", end=end)
    if len(kargs) > 0:
        for key, item in kargs.items():
            print(key, end=": ")
            print(item, end="")
        print("", end=end)


def p(*s, end="\n", **kargs):
    if tconfig.TUTILS_INFO or tconfig.TUTILS_DEBUG or tconfig.TUTILS_WARNING:
        print("[Trans Info] ", end="")
        tprint(*s, end="\n", **kargs)


def w(*s, end="\n", **kargs):
    if tconfig.TUTILS_WARNING or tconfig.TUTILS_DEBUG:
        print("[Trans Warning] ", end="")
        tprint(*s, end="\n", **kargs)


def d(*s, end="\n", **kargs):
    if tconfig.TUTILS_DEBUG:
        print("[Trans Debug] ", end="")
        tprint(*s, end="\n", **kargs)
