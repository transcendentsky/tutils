# coding: utf-8
from .learner import LearnerModule
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# trans utils
from tutils.tutils.ttimer import tenum, timer
from tutils.tutils import tfilename, CSVLogger
from torch.cuda.amp import autocast, GradScaler
from .recorder import Recorder
from tqdm import tqdm
from datetime import datetime


class Trainer(object):
    def __init__(self, 
                logger=None, 
                config=None, 
                mode="ps", 
                runs_dir=None, 
                tag=None,
                num_epochs=1200, 
                batch_size=4, 
                num_workers=4, 
                save_interval=50, 
                use_amp=False, 
                val_check_interval=100,
                tester=None, 
                monitor=None, 
                save_latest_only=False, 
                training_log_interval=1,         # training log every n epochs
                gpus=4, 
                sync_batchnorm=True, 
                **kwargs):
        """
            mode:
                ps:  Parameter Server
                ddp: Distributed data parallel
        """
        assert mode in ['ps', 'ddp', 'autotune']
        self.mode = mode
        if mode == 'autotune': 
            try:
                from ray import tune
            except:
                raise EnvironmentError
        if mode == "ddp": raise NotImplementedError #self.init_ddp_env()
        else: self.device = torch.device("cuda")
        self.init_timers()
        # Config
        self.config = config
        self.tag = tag # config['tag']
        self.runs_dir = runs_dir # config['runs_dir']
        self.max_epochs = num_epochs # config['training']['num_epochs']
        self.batch_size = batch_size # config["training"]['batch_size']
        self.num_workers = num_workers # config['training']['num_workers']
        self.save_interval = save_interval # config['training']['save_interval']
        self.use_amp = use_amp # config['training']['use_amp']
        self.val_check_interval = val_check_interval # config['validation']['val_check_interval']
        self.save_latest_only = save_latest_only
        self.training_log_interval = training_log_interval
        if self.use_amp:
            self.logger.info("[*] You are using AMP accelaration. ")

        # Other
        self.recorder = Recorder()
        self.recorder_test = Recorder()
        self.logger = logger if logger is not None else FakeLogger()
        self.csvlogger = CSVLogger(tfilename(runs_dir, "best_record"))

        self.monitor = monitor
        self.tester = tester
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # initialize in "init_ps"

    def init_timers(self):
        self.timer_epoch = timer("one epoch")
        self.timer_data  = timer("data time")
        self.timer_net   = timer("net forwarding")
        self.timer_eval  = timer("evaluation")
        self.timer_write = timer("writing files")

    def init_model(self, model, trainset, valset=None):
        if self.mode == "ddp":
            raise NotImplementedError("to Device is not Implemented")
            # return self.init_ddp(model, trainset, valset)
        elif self.mode == "ps":
            return self.init_ps(model, trainset, valset)
        else:
            raise NotImplementedError(f"mode={self.mode}")

    def init_ps(self, model, trainset, valset):
        assert len(trainset) > 0 , f"Got {len(trainset)}"
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, drop_last=True)
        if valset is not None:
            assert len(valset) > 0 , f"Got {len(valset)}"
            valloader = DataLoader(dataset=valset,
                                 batch_size=1, num_workers=1)
        else:
            valloader = None

        model.net = torch.nn.DataParallel(model.net)
        model.load()
        model.cuda()

        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("--------------------------\n "
                             "      [*] Using AMP !\n"
                             " -------------------------")

        return model, trainloader, valloader

    def fit(self, model, trainset, valset=None):
        assert isinstance(model, LearnerModule)
        model, trainloader, valloader = self.init_model(model, trainset)

        # Set optimizer and scheduler
        optim_configs = model.configure_optimizers()
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        for epoch in range(self.max_epochs):
            # Training
            self.train(model, trainloader, epoch, optimizer, scheduler)
            # Evaluation
            if epoch % self.val_check_interval == 0:
                if valset is not None:
                    self.validate(model, valset, epoch)
                if self.tester is not None:
                    out = self.tester.test(model, epoch)
                    if self.monitor is not None:
                        best_dict = self.monitor.record(out, epoch)
                        self.recorder_test.record({**best_dict, **out})
                        if best_dict['isbest']:
                            self.save_best_model(model, epoch)
                            self.csvlogger.record({**best_dict, **out, "time":_get_time_str()})
                        self.logger.info(f"\n[*] {dict_to_str(best_dict)}[*] Epoch {epoch}: \n{dict_to_str(out)}")
                        self.logger.add_scalars(out, step=epoch, tag='test')
                    else:
                        self.logger.info(f"\n[*] Epoch {epoch}: {dict_to_str(out)}")
            self.autosave(model, epoch)

    def on_before_zero_grad(self, **kwargs):
        pass

    def on_after_training(self, **kwargs):
        pass

    def train(self, model, trainloader, epoch, optimizer, scheduler=None):
        self.on_before_zero_grad()
        model.train()
        do_training_log = (epoch % self.training_log_interval == 0)
        
        self.recorder.clear()
        time_record = 0.1111
        self.timer_epoch()
        for load_time, batch_idx, data in tqdm(tenum(trainloader), ncols=100):
            model.on_before_zero_grad()
            optimizer.zero_grad()
            self.timer_data()
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.cuda()
            time_data_cuda = self.timer_data()
            if self.use_amp:
                with autocast():
                    self.timer_net()
                    out = model.training_step(data, batch_idx)
                    time_fd = self.timer_net()
                    loss = out['loss']
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    time_bp = self.timer_net()
            else:                
                self.timer_net()
                out = model.training_step(data, batch_idx)
                time_fd = self.timer_net()
                loss = out['loss']
                loss.backward()
                optimizer.step()
                time_bp = self.timer_net()
            
            if torch.isnan(loss):
                raise ValueError(" loss got Nan !!! ")

            out['time_load'] = load_time
            out['time_cuda'] = time_data_cuda
            out['time_forward'] = time_fd
            out['time_bp'] = time_bp
            out['time_record'] = time_record

            if do_training_log:
                self.timer_data()
                self.recorder.record(out)
                time_record = self.timer_data()

            if epoch == 0:
                self.logger.info("[*] Debug Checking Pipeline !!!")
                break

        lr = optimizer.param_groups[0]['lr']

        _dict = None
        if do_training_log:
            _dict = self.recorder.cal_metrics()
            _dict['time_total'] = self.timer_epoch()
            _dict['lr'] = lr
            # print(_dict)
            # assert isinstance(lr, float), f"Got lr={lr}, type: {type(lr)}"
            loss_str = ""
            for k, v in _dict.items():
                loss_str += "{}:{:.3f} ".format(k, v)
            self.logger.info(f"Epoch {epoch}: {loss_str}")            
            self.logger.add_scalars(_dict, step=epoch, tag='train')
        
        self.on_after_training(d=_dict)

    def validate(self, model, valloader, epoch):
        model.eval()
        with torch.no_grad():
            self.recorder.clear()
            for load_time, batch_idx, data in tenum(valloader):
                self.timer_eval()
                for k, v in data.items():
                    if type(v) == torch.Tensor:
                        data[k] = v.cuda()
                out = model.validation_step(data, batch_idx)
                time_eval = self.timer_eval()
                out['load_time'] = load_time
                out['time_eval'] = time_eval
                self.recorder.record(out)

        _dict = self.recorder.cal_metrics()
        loss_str = ""
        for k, v in _dict.items():
            loss_str += "{}:{:.3f} ".format(k, v) # f"{v}:{loss_values[i]}; "
        self.logger.info(f"\n\tValidation step, Epoch {epoch}: {loss_str}")            
        self.logger.add_scalars(_dict, step=epoch, tag='val')
        return loss_values

    def _init_test(self, model):
        model.net = torch.nn.DataParallel(model.net)
        model.load()
        model.cuda()
        return model

    def test(self, model):
        # """
        #     Use Tester.test instead
        # """
        # raise NotImplementedError
        model = self._init_test(model)
        assert self.tester is not None, "No Tester !!!"
        out = self.tester.test(model, 0)
        self.csvlogger.record({**out, "time":_get_time_str()})
        self.logger.info(f"\n[*] Results: \n{dict_to_str(out)}")
        

    def autosave(self, model, epoch):
        if self.save_interval > 0 and epoch % self.save_interval == 0:
            if self.save_latest_only:
                return self.save_latest(model, epoch)
            return self.save(model, epoch)

    def save_latest(self, model, epoch):        
        model.save(tfilename(self.runs_dir + "/ckpt/model_latest.pth"), epoch=epoch, is_latest=True)
        self.logger.info(f"Epoch {epoch}: Just Saved model to ``{self.runs_dir + '/ckpt/model_latest.pth'}``! ")

    def save(self, model, epoch):
        model.save(tfilename(self.runs_dir + "/ckpt/model_epoch_{}.pth".format(epoch)), epoch=epoch)
        self.logger.info(f"Epoch {epoch}: Just Saved model to ``{self.runs_dir + '/ckpt/model_epoch_{}.pth'.format(epoch)}``! ")

    def save_best_model(self, model, epoch):        
        model.save(tfilename(self.runs_dir + "/ckpt/best_model_epoch_{}.pth".format(epoch)), epoch=epoch, is_best=True)
        self.logger.info(f"[Best model] Epoch {epoch}: Saved best model to ``{self.runs_dir + '/ckpt/model_epoch_{}.pth'.format(epoch)}``! ")

    # def init_ddp_env(self):
    #     # 1 Initialize
    #     torch.distributed.init_process_group(backend="nccl")
    #     # 2 set up gpu for each process
    #     self.local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(self.local_rank)
    #     self.device = torch.device("cuda", self.local_rank)
    #     print("debug: ", self.local_rank, self.device)
    #
    # def init_ddp(self, model, trainset, valset):
    #     # 3 Use DistributedSampler
    #     trainloader = DataLoader(dataset=trainset,
    #                              batch_size=self.batch_size,
    #                              sampler=DistributedSampler(trainset), shuffle=True, drop_last=True)
    #     # 4 move the network to the predetermined gpu
    #     model.to(self.device)
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         # 5) pack up with DistributedDataParallel
    #         model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                           device_ids=[self.local_rank],
    #                                                           output_device=self.local_rank)
    #     return model, trainloader
    #
    # def cleanup(self):
    #     torch.distributed.destroy_process_group()

def dict_to_str(d):
    loss_str = ""
    for k, v in d.items():
        loss_str += "\t {}\t: {} \n".format(k, v) # f"{v}:{loss_values[i]}; "
    return loss_str

def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')

class FakeLogger:
    def __init__(self) -> None:
        pass

    def info(self, msg, *args, **kwargs):
        print(msg, *args, **kwargs)

    def add_scalar(self, msg, *args, **kwargs):
        pass

if __name__ == '__main__':
    # ------------  For debug  --------------
    class RandomDataset(Dataset):
        """ Just For Testing"""
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size).to('cuda')

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len


    class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super(Model, self).__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, input):
            output = self.fc(input)
            print("  In Model: input size", input.size(),
                  "output size", output.size())
            return output

    model = Model(5, 2)
    dataset = RandomDataset(5, 90)
    trainer = Trainer(config={"training":{"batch_size":256, "num_epochs":500}})
    trainer.fit(model, dataset)