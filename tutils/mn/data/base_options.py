# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Common
        self.parser.add_argument("--outputs_dir", type=str, default="./outputs", help="models are saved here")
        self.parser.add_argument("--datadir", type=str, default=None)
        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--checkpoint", type=str, default=None, help="models are saved here")  
        self.parser.add_argument("--lr", type=float, default=0.001)
        self.parser.add_argument("--weight_decay", type=float, default=0.005)
        
        self.parser.add_argument("--istrain", action="store_true")
        self.parser.add_argument("--name", type=str, default="model")
        # self.parser.add_argument()
        # self.parser.add_argument()
        # self.parser.add_argument()


        # experiment specifics
        self.parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        self.parser.add_argument("--model", type=str, default="pix2pixHD", help="which model to use")
        self.parser.add_argument("--norm", type=str, default="instance", help="instance normalization or batch normalization")
        self.parser.add_argument("--use_dropout", action="store_true", help="use dropout for the generator")
        self.parser.add_argument("--data_type", type=int, default=32, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit",)
        self.parser.add_argument("--verbose", action="store_true", default=False, help="toggles verbose")     
        self.parser.add_argument("--nThreads", default=2, type=int, help="# threads for loading data")
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.opt.gpu_ids.append(int_id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            # pass
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.outputs, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, "opt.txt")
            with open(file_name, "wt") as opt_file:
                opt_file.write("------------ Options -------------\n")
                for k, v in sorted(args.items()):
                    opt_file.write("%s: %s\n" % (str(k), str(v)))
                opt_file.write("-------------- End ----------------\n")
        return self.opt

