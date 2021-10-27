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
from tutils.trainer.learner import LearnerModule
from tutils.trainer.recorder import Recorder
from tqdm import tqdm
from datetime import datetime
import os
import signal
import subprocess
import sys
from .utils.trainer_utils import MultiOptimizer, MultiScheduler


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


class DDPWorker(object):
    def __init__(self,
                 logger,
                 config,
                 rank,
                 world_size,
                 tester,
                 monitor):
        # self.logger = logger[0] if rank == 0 else None
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.tester = tester

        self.gpus = config['training'].get('gpus', '0,1,2,3,4,5,6,7')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        # print("[*] GPUs: ", self.gpus)

        self.tag = config['base']['tag']
        self.runs_dir = config['base']['runs_dir']
        self.max_epochs = config['training'].get('num_epochs', 400)
        self.batch_size = config["training"]['batch_size']
        self.num_workers = config['training']['num_workers']
        self.save_interval = config['training']['save_interval']

        self.load_pretrain_model = config['training'].get('load_pretrain_model', False)
        self.pretrain_model = config['training'].get('pretrain_model', None)
        self.val_check_interval = config['training'].get('val_check_interval', 50)
        self.training_log_interval = config['training'].get('training_log_interval', 1)
        self.use_amp = config['training'].get('use_amp', False)
        self.save_latest_only = config['training'].get('save_latest_only', False)

        self.init_timers()

        # Logging, in GPU 0
        if self.rank == 0:
            print("Logger at Process(rank=0)")
            self.recorder = Recorder()
            self.recorder_test = Recorder()
            self.logger = None
            self.csvlogger = CSVLogger(tfilename(self.runs_dir, "best_record"))
            self.monitor = monitor
            self.tester = tester

        self.optimizer = None
        self.scheduler = None

        self.ddp_config = config['training'].get('ddp', dict())
        self.master_addr = self.ddp_config.get('master_addr', 'localhost')
        self.master_port = str(self.ddp_config.get('master_port', '25700'))
        self.batch_size = config['training'].get('batch_size', 4)
        self.num_worker = config['training'].get('num_worker', 2)
        self.world_size = torch.cuda.device_count()
        self.dist_url = 'tcp://' + self.master_addr + ":" + self.master_port

        torch.distributed.init_process_group(backend="nccl", init_method=self.dist_url,
            world_size=self.world_size, rank=self.rank)

        self.scaler = GradScaler() if self.use_amp else None

    def init_timers(self):
        self.timer_epoch = timer("one epoch") if self.rank == 0 else VoidTimer()
        self.timer_batch = timer("a batch") if self.rank == 0 else VoidTimer()
        self.timer_data = timer("data time") if self.rank == 0 else VoidTimer()
        self.timer_net = timer("net forwarding") if self.rank == 0 else VoidTimer()
        self.timer_eval = timer("evaluation") if self.rank == 0 else VoidTimer()
        self.timer_write = timer("writing files") if self.rank == 0 else VoidTimer()

    def fit(self, model, trainset):
        if self.rank == 0:
            assert isinstance(model, LearnerModule), "model type error!"
            self.logger = model.configure_logger()['logger']

        # Initialize Models and DataLoader and Optimizers
        if self.load_pretrain_model:
            model.load()
        model.net = model.net.to(self.rank)
        model.net = nn.SyncBatchNorm.convert_sync_batchnorm(model.net)
        model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[self.rank], find_unused_parameters=True)

        sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        assert self.batch_size % self.world_size == 0
        per_device_batch_size = self.batch_size // self.world_size
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=per_device_batch_size, num_workers=self.num_worker,
            pin_memory=True, sampler=sampler, drop_last=True)

        optimizer, scheduler = self.configure_optim(model)

        for epoch in range(self.max_epochs):
            self.on_before_zero_grad()
            # Training
            do_training_log = (epoch % self.training_log_interval == 0)
            self.train(model, self.trainloader, epoch, optimizer, scheduler, do_training_log)

            # epoch logger !
            if do_training_log and self.rank == 0:
                _dict = self.recorder.cal_metrics()
                _dict['time_total'] = self.timer_epoch()
                # print(_dict)
                # assert isinstance(lr, float), f"Got lr={lr}, type: {type(lr)}"
                loss_str = ""
                for k, v in _dict.items():
                    loss_str += "{}:{:.4f} ".format(k, v)
                # lr = optimizer.param_groups[0]['lr']
                lr = self.get_lr(optimizer)
                _dict['lr'] = lr
                loss_str += "{}:{:.6e} ".format('lr', lr)
                self.logger.info(f"Epoch {epoch}: {loss_str}")
                self.logger.add_scalars(_dict, step=epoch, tag='train')

            # Evaluation
            if epoch % self.val_check_interval == 0 and self.rank == 0:
                print("Note: Tester runs on <rank 0> only")
                if self.tester is not None:
                    out = self.tester.test(model, epoch, self.rank)
                    if self.monitor is not None:
                        best_dict = self.monitor.record(out, epoch)
                        self.recorder_test.record({**best_dict, **out})
                        if best_dict['isbest']:
                            self.save(model, epoch, type='best')
                            self.csvlogger.record({**best_dict, **out, "time": _get_time_str()})
                        self.logger.info(f"\n[*] {dict_to_str(best_dict)}[*] Epoch {epoch}: \n{dict_to_str(out)}")
                        self.logger.add_scalars(out, step=epoch, tag='test')
                    else:
                        self.logger.info(f"\n[*] Epoch {epoch}: {dict_to_str(out)}")
                self.save(model, epoch)
        print("Training is Over for GPU rank ", self.rank)
        self.cleanup()

    def configure_optim(self, model, **kwargs):
        # Set optimizer and scheduler
        optim_configs = model.configure_optimizers()
        assert isinstance(optim_configs, dict)
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        if isinstance(optimizer, list):
            optimizer = MultiOptimizer(optimizer)
        if isinstance(scheduler, list):
            scheduler = MultiScheduler(scheduler)
        return optimizer, scheduler


    def on_before_zero_grad(self, **kwargs):
        pass

    def train(self, model, trainloader, epoch, optimizer, scheduler=None, do_training_log=True):
        model.train()

        if do_training_log and self.rank == 0:
            self.recorder.clear()
            time_record = 0.1111
            self.timer_batch()

        for load_time, batch_idx, data in tenum(trainloader):
            model.on_before_zero_grad()
            optimizer.zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            if self.use_amp:
                with autocast():
                    self.timer_net()
                    out = model.training_step(data, batch_idx)
                    assert isinstance(out, dict)
                    time_fd = self.timer_net()
                    loss = out['loss']
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    time_bp = self.timer_net()
            else:
                self.timer_net()
                out = model.training_step(data, batch_idx)
                if torch.isnan(out['loss']):
                    print("Nan Value: ", out['loss'])
                    raise ValueError
                assert isinstance(out, dict)
                time_fd = self.timer_net()
                loss = out['loss']
                loss.backward()
                optimizer.step()
                time_bp = self.timer_net()

            time_batch = self.timer_batch()
            # batch logger !
            if do_training_log and self.rank == 0:
                out['time_load'] = load_time
                out['time_cuda'] = time_data_cuda
                out['time_forward'] = time_fd
                out['time_bp'] = time_bp
                out['time_record'] = time_record
                out['time_batch'] = time_batch
                self.timer_data()
                self.recorder.record(out)
                time_record = self.timer_data()
            # for debug !
            if epoch == 0:
                if self.rank == 0:
                    self.logger.info("[*] Debug Checking Pipeline !!!")
                break
        scheduler.step()

    def save(self, model, epoch, type=None):
        if self.rank != 0:
            return
        if type is None:
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                save_name = "/ckpt/model_epoch_{}.pth".format(epoch)
                model.save(tfilename(self.runs_dir, save_name), epoch=epoch)
                self.logger.info(f"Epoch {epoch}: Save model to ``{save_name}``! ")
        elif type == 'latest':
            save_name = "/ckpt/model_latest.pth"
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                if self.save_latest_only:
                    model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_latest=True)
                    self.logger.info(f"Epoch {epoch}: Save model to ``{save_name}``! ")
        elif type == 'best':
            save_name = "/ckpt/best_model_epoch_{}.pth".format(epoch)
            model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_best=True)
            self.logger.info(f"[Best model] Epoch {epoch}: Save model to ``{save_name}``! ")
        elif type == "training_stat":
            save_name = "/ckpt/model_latest.pth"
            model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_latest=True)
            save_optim_name = "/ckpt/optim_latest.pth"
            model.save_optim(tfilename(self.runs_dir, save_optim_name), optimizer=self.optimizer, epoch=epoch)
            self.logger.info(f"Epoch {epoch}: Save checkpoint to ``{save_name}``")

    def cleanup(self):
        torch.distributed.destroy_process_group()

    def info(self, msg, *args, **kwargs):
        if self.rank == 0:
            self.logger.info(msg, *args, **kwargs)

    def get_lr(self, optimizer):
        if isinstance(optimizer, MultiOptimizer):
            return optimizer.get_lr()
        else:
            return optimizer.param_groups[0]['lr']


class VoidTimer(object):
    def __init__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        return None

# def handle_sigusr1(signum, frame):
#     os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
#     exit()
#
#
# def handle_sigterm(signum, frame):
#     pass


def dict_to_str(d):
    loss_str = ""
    for k, v in d.items():
        loss_str += "\t {}\t: {} \n".format(k, v)  # f"{v}:{loss_values[i]}; "
    return loss_str


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')




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