"""
This is a template for Tester
"""

import torch
import numpy as np
from tutils import tfunctime
from typing import Dict, List, Tuple
import time



class Recorder(object):
    def __init__(self, logger=None, config=None):
        super(Recorder, self).__init__()
        self.logger = logger
        self.config = config
        self.loss_list = []
        self.loss_keys = None

    def clear(self):
        self.loss_list.clear()

    def record(self, loss:dict) -> None:
        assert type(loss) == dict, f"Got {loss}"
        # print("debug record", loss)
        if self.loss_keys is None:
            self.loss_list = []
            self.loss_keys = loss.keys()
        l_list = []
        for key, value in loss.items():
            if type(value) == torch.Tensor:
                l_list.append(value.detach().cpu().item())
            elif type(value) in [str, bool]:
                pass
            elif type(value) in [np.ndarray, np.float64, np.float32, int]:
                l_list.append(float(value))
            else:
                print("debug??? type Error? , got ", type(value))
                l_list.append(float(value))
        self.loss_list.append(l_list)

    def _record(self, loss):
        if type(loss) == torch.Tensor:
            loss = loss.detach().cpu().item()
            if self.loss_list is None:
                self.loss_list = []
            self.loss_list.append(loss)
        elif type(loss) == list:
            for i, lossi in enumerate(loss):
                if type(lossi) == torch.Tensor:
                    loss[i] = lossi.detach().cpu().item()
            if self.loss_list is None:
                self.loss_list = []
            self.loss_list.append(loss)
        elif type(loss) == dict:
            if self.loss_list is None:
                self.loss_list = []
                self.loss_keys = loss.keys
            l_list = []
            for lossi, value in loss.items():
                if type(lossi) == torch.Tensor:
                    l_list.append(lossi.detach().cpu().item())
            self.loss_list.append(l_list)

    def cal_metrics(self):
        if len(self.loss_list) <= 0:
            return None
        temp = np.array(self.loss_list)
        mean = temp.mean(axis=0)
        # print("debug mean", mean, temp, self.loss_keys)
        return mean, self.loss_keys


class EpochLogger:
    def __init__(self, iter, logger, config, mode="dec"):
        # mode in ["dec", "inc"]
        self.logger = logger
        self.config = config
        self.best_epoch = None
        self.best_value = None
        self.epoch_list = []
        self.value_list = []
        self.mode = mode
        if mode == "dec":
            self.comp = self._less
        else:
            self.comp = self._greater

        self.stop_time = time.time()

        if type(iter) != enumerate:
            self.iter = enumerate(iter)
        else:
            self.iter = iter

        self.recorder = Recorder()

    def _less(self, a, b):
        return a < b

    def _greater(self, a, b):
        return a > b

    def record_and_return(self, epoch, value):
        if self.best_epoch is None:
            self.best_epoch = epoch
            self.best_value = value
        if self.comp(value, self.best_epoch):
            self.best_epoch = epoch
            self.best_value = value
            return True
        else:
            return False

    def __iter__(self):
        return self

    def __next__(self):
        self.recorder.cal_metrics()
        start = time.time()
        i, res = self.iter.__next__()
        # clear
        self.recorder.clear()
        return self.recorder, res



class BatchLogger:
    def __init__(self, iter, logger, config, mode="dec"):
        # mode in ["dec", "inc"]
        self.logger = logger
        self.config = config
        self.best_epoch = None
        self.best_value = None
        self.epoch_list = []
        self.value_list = []
        self.mode = mode
        if mode == "dec":
            self.comp = self._less
        else:
            self.comp = self._greater

        self.stop_time = time.time()
        if type(iter) != enumerate:
            self.iter = enumerate(iter)
        else:
            self.iter = iter

        self.recorder = Recorder()

    def _less(self, a, b):
        return a < b
        # if a < b: return True
        # else: return False

    def _greater(self, a, b):
        return a > b
        # if a > b: return True
        # else: return False

    def record_and_return(self, epoch, value):
        if self.best_epoch is None:
            self.best_epoch = epoch
            self.best_value = value
        # self.best_epoch.append(epoch)
        # self.best_value.append(value)
        if self.comp(value, self.best_epoch):
            self.best_epoch = epoch
            self.best_value = value
            return True
        else:
            return False

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        i, res = self.iter.__next__()
        end = time.time()
        iter_time = end - self.stop_time
        self.stop_time = end
        # clear
        self.recorder.clear()
        return (end-start), self.recorder, i, res


class Tester(object):
    def __init__(self, args, config, logger):
        self.args = args
        self.config = config
        self.logger = logger
        self.evaluater = Recorder(logger, config)
        self.model = None

    def test(self, net=None, epoch=None, train="", dump_label=False):
        self.evaluater.reset()
        self.model = net
        """
        ...
        """
        mre = self.evaluater.cal_metrics()
        return mre


