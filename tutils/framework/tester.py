"""
This is a template for Tester
"""

import torch
import numpy as np
from tutils import tfunctime


class Recoder(object):
    def __init__(self, logger,config):
        super(Recoder, self).__init__()
        self.logger = logger
        self.config = config
        self.loss_list = []

    def clear(self):
        self.loss_list.clear()

    def record(self, loss):
        if type(loss) == torch.Tensor:
            loss = loss.detach().cpu().item()
        elif type(loss) == list:
            for i, lossi in enumerate(loss):
                if type(lossi) == torch.Tensor:
                    loss[i] = lossi.detach().cpu().item()
        self.loss_list.append(loss)
        
    def cal_metrics(self):
        temp = np.array(self.loss_list)
        mean = temp.mean(axis=0)
        # self.logger.info(f"cal_metrics: {mean}")
        return mean

class EpochRecorder(object):
    def __init__(self, logger, config, mode="dec"):
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

    def _less(self, a, b):
        if a < b: return True
        else: return False

    def _greater(self, a, b):
        if a > b: return True
        else: return False


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


class Tester(object):
    def __init__(self, args, config, logger):
        self.args = args
        self.config = config
        self.logger = logger
        self.evaluater = Recoder(logger, config)
        self.model = None

    def test(self, net=None, epoch=None, train="", dump_label=False):
        self.evaluater.reset()
        self.model = net
        """
        ...
        """
        mre = self.evaluater.cal_metrics()
        return mre

# class Evaluater(object):
#     def __init__(self, logger, config, metric=BaseMetric(), loss_num=BaseMetric().loss_num):
#         self.logger = logger
#         self.config = config
#         self.metric = metric
#         self.loss_num = 1
#         if loss_num == 1:
#             self.loss_list_list = None
#             self.loss_list = []
#         elif loss_num > 1:
#             self.loss_list_list = [[] for i in range(loss_num)]
#         else:
#             raise ValueError

#     def reset(self):
#         if self.loss_list_list == None:
#             self.loss_list.clear()
#         else:
#             for loss_list in self.loss_list_list:
#                 loss_list.clear()

#     def record(self, pred, gt):
#         if self.loss_num == 1:
#             loss = self.metric(pred, gt).cpu().item()
#             self.loss_list.append(loss)
            
#     def cal_metrics(self):
#         if self.loss_num == 1:
#             temp = np.array(self.loss_list)
#             mean = temp.mean()
#             self.logger.info(mean)
#             return mean
#         else:
#             # calculate MRE SDR
#             temp = np.array(self.loss_list_list)
#             mean = temp.mean(axis=0)
#             self.logger.info(mean)
#             return mean