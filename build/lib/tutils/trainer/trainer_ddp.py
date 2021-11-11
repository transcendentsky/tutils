# coding: utf-8
"""
    Borrow some code and ideas from:
        https://github.com/facebookresearch/barlowtwins

"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# trans utils
from tutils.tutils.ttimer import tenum, timer
from tutils.tutils import tfilename, CSVLogger, MultiLogger
from torch.cuda.amp import autocast, GradScaler
from tutils.trainer.learner_module import LearnerModule
import os
from .trainer_abstract import AbstractTrainer


def ddptrainer_from_config(config, tester, monitor, **kwargs):
    return DDPTrainer(config=config,
                      tester=tester,
                      monitor=monitor,
                      kwargs=kwargs)


class DDPTrainer(object):
    def __init__(self,
                 logger=None,
                 config=None,
                 tester=None,
                 monitor=None,
                 mode="ddp",
                 **kwargs
                 ):
        """
            mode:
                ps:  Parameter Server
                ddp: Distributed data parallel
        """
        assert mode in ['ps', "ddp"]
        # self.logger = [logger]
        self.logger = None
        # assert logger is None
        self.mode = mode
        self.config = config
        self.tester = tester
        self.monitor = monitor
        self.gpus = config['training'].get('gpus', '0,1,2,3,4,5,6,7')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        print("[*] GPUs: ", self.gpus)
        self.world_size = torch.cuda.device_count()

    def fit(self, model, trainset, valset=None, **kwargs):
        return self.init_ddp(model, trainset, **kwargs)

    def init_ddp(self, model, trainset, **kwargs):
        torch.multiprocessing.spawn(start_ddp_worker,
                                    (self.world_size, self.logger, self.config, model, trainset, self.tester, self.monitor),
                                    nprocs=self.world_size,
                                    join=True, )


def start_ddp_worker(rank, world_size, logger, config, model, trainset, tester, monitor):
    print(f"Start DDP worker: rank={rank}, world_size={world_size}")
    worker = DDPWorker(logger=logger,
                       config=config,
                       rank=rank,
                       world_size=world_size,
                       tester=tester,
                       monitor=monitor)
    worker.fit(model, trainset)
    pass


class DDPWorker(AbstractTrainer):
    def __init__(self,
                 logger,
                 config,
                 rank,
                 world_size,
                 tester,
                 monitor):
        super(DDPWorker, self).__init__(config, tester, monitor, rank, world_size)

    def init_model(self, model, trainset, **kwargs):
        if self.rank == 0:
            assert isinstance(model, LearnerModule), "model type error!"
            self.logger = model.configure_logger()['logger']

        # Initialize Models and DataLoader and Optimizers
        if self.load_pretrain_model:
            model.load()
        model.net = model.net.to(self.rank)
        model.net = nn.SyncBatchNorm.convert_sync_batchnorm(model.net)
        model.net = torch.nn.parallel.DistributedDataParallel(model.net,
                                                              device_ids=[self.rank],
                                                              find_unused_parameters=True)

        sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        assert self.batch_size % self.world_size == 0
        per_device_batch_size = self.batch_size // self.world_size
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=per_device_batch_size, num_workers=self.num_workers,
            pin_memory=True, sampler=sampler, drop_last=True)
        return model



if __name__ == '__main__':
    from tutils import load_yaml, trans_args, trans_init, print_dict
    from tutils.trainer import Monitor
    import argparse
    # ------------  For debug  --------------

    
    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    class Learner(LearnerModule):
        def __init__(self, config, logger=None):
            super(Learner, self).__init__(config)
            self.config = config
            self.net = ToyModel()
            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            return self.net(x)

        def training_step(self, data, batch_idx, **kwargs):
            imgs, labels = data['imgs'], data['labels']
            outputs = self.forward(imgs) # torch.randn(20, 10)
            loss = self.loss_fn(outputs, labels)
            return {'loss':loss}

        def configure_optimizers(self, **kwargs):
            optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
            scheduler = None
            return {'optimizer': optimizer, "scheduler": scheduler}

        def configure_logger(self):
            logger = MultiLogger(logdir=self.config['base']['runs_dir'],
                                mode=self.config['logger']['mode'],
                                tag=self.config['base']['tag'],
                                extag=self.config['base'].get('extag', None),
                                action=self.config['logger'].get('action', 'k'))
            return {'logger': logger}

    class RandomDataset(Dataset):
        """ Just For Testing"""

        def __init__(self, emb, datalen):
            self.len = datalen
            self.data = torch.randn(datalen, emb)
            self.label = torch.randn(datalen, 5)

        def __getitem__(self, index):
            return {"imgs": self.data[index], "labels": self.label[index]}

        def __len__(self):
            return self.len

    class Tester(object):
        def __init__(self):
            pass

        def test(self, model, epoch, rank, **kwargs):
            sample = torch.randn(10, 10).to(rank)
            output = model(sample)
            # metric
            return {"rrres": float(epoch), "loss":0.2}


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/config.yaml")
    args = trans_args(parser)
    logger, config = trans_init(args)
    print_dict(config)

    monitor = Monitor(key='rrres', mode='inc')
    tester = Tester()
    # import ipdb; ipdb.set_trace()
    # model = ToyModel()
    model = Learner(config)
    dataset = RandomDataset(10, 100)
    trainer = DDPTrainer(logger, config, tester=tester, monitor=monitor)
    trainer.fit(model, dataset)