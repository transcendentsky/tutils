from collections import OrderedDict
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


def tfuncname(func):
    def run(*argv, **kargs):
        # p("--------------------------------------------")
        d("[Trans Utils] Function Name: ", end=" ")
        d(func.__name__)
        ret = func(*argv, **kargs)
        # if argv:
        #     ret = func(*argv)
        # else:
        #     ret = func()
        return ret
    return run

def time_now():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def generate_random_str(n: int = 6):
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, n))
    return ran_str


def generate_name():
    return time_now() + '-' + generate_random_str(6)


# def write_image_np(image, filename):
#     cv2.imwrite("wc_" + generate_random_str(5)+'-'+ time_now() +".jpg", image.astype(np.uint8))
#     pass

def tdir(*dir_paths):
    def checkslash(name):
        if name.startswith("/"):
            name = name[1:]
            return checkslash(name)
        else:
            return name
    if len(dir_paths) <= 1:
        return dir_paths[0]
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
        return filenames[0]
    names = [filenames[0]]
    for name in filenames[1:]:
        names.append(checkslash(name))
    filename = os.path.join(*names)
    d(filename)
    parent, name = os.path.split(filename)
    if not os.path.exists(parent):
        d(parent)
        os.makedirs(parent)
    return filename


def texists(*filenames):
    path = os.path.join(*filenames)
    return os.path.exists(path)


# def ttsave(state, path, configs=None):
#     path = tdir("trans_torch_models", path, generate_name())
#     if configs is not None:
#         assert type(configs) is dict
#         config_path = tfilename(path, "configs.json")
#         with open(config_path, "wb+") as f:
#             config_js = json.dumps(configs)
#             f.write(config_js)
#     torch.save(state, tfilename(path, "model.pth"))


def add_total(tuple1, tuple2):
    l = list()
    for i, item in enumerate(tuple1):
        l.append(tuple1[i] + tuple2[i])
    return tuple(l)


if __name__ == "__main__":
    d(adadada="tutils")
