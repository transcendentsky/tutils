# coding: utf-8
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# trans utils
from tutils.tutils.ttimer import timer
from tutils.tutils import tfilename
from torch.cuda.amp import autocast, GradScaler
from .recorder import Recorder
from tqdm import tqdm
from .excel_recorder import CSVLogger
from datetime import datetime


class Monitor(object):
    def __init__(self, key, mode="inc"):
        """ mode = inc or dec """
        self.mode = mode
        assert mode in ['inc', 'dec']
        self.best_epoch = None
        self.best_value = None
        self.key = key
        self.best_dict = None

    def is_better(self, v):
        if self.mode == "inc":
            return v > self.best_value
        else:
            return v < self.best_value

    def record(self, d, epoch):
        isbest = self._record(d[self.key], epoch)
        if isbest:
            print("[Monitor] `Achive New Record` ")
            self.best_dict = d
        return {"isbest":isbest, "best_value":self.best_value, "best_epoch":self.best_epoch, **self.best_dict}

    def _record(self, v, epoch):
        if self.best_epoch is None or self.best_value is None:
            self.best_value = v
            self.best_epoch = epoch
            return True
        if self.is_better(v):
            self.best_value = v
            self.best_epoch = epoch
            return True
        else:
            return False


def dict_to_str(d):
    loss_str = ""
    for k, v in d.items():
        loss_str += "{}:{} ".format(k, v) # f"{v}:{loss_values[i]}; "
    return loss_str

def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')

class Trainer(object):
    def __init__(self, logger=None, config=None, mode="ps", runs_dir=None, tag=None,
                num_epochs=1200, batch_size=4, num_workers=4, save_seq=50, use_amp=False, val_seq=100,
                tester=None, monitor=None, gpus=4, sync_batchnorm=True, save_latest_only=False, **kwargs):
        """
            mode:
                ps:  Parameter Server
                ddp: Distributed data parallel
        """
        assert mode in ['ps', "ddp"]
        self.mode = mode
        if mode == "ddp": raise NotImplementedError #self.init_ddp_env()
        else: self.device = torch.device("cuda")
        self.logger = logger
        self.init_timers()
        # Config
        self.config = config
        self.tag = tag # config['tag']
        self.runs_dir = runs_dir # config['runs_dir']
        self.max_epochs = num_epochs # config['training']['num_epochs']
        self.batch_size = batch_size # config["training"]['batch_size']
        self.num_workers = num_workers # config['training']['num_workers']
        self.save_seq = save_seq # config['training']['save_seq']
        self.use_amp = use_amp # config['training']['use_amp']
        self.val_seq = val_seq # config['validation']['val_seq']
        self.save_latest_only = save_latest_only

        # Other
        self.recorder = Recorder()
        self.recorder_test = Recorder()
        self.csvlogger = CSVLogger(tfilename(runs_dir, "csv"))
        self.monitor = monitor
        self.tester = tester
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # initialize in "init_ps"

    def init_timers(self):
        self.timer1 = timer("data loading")
        self.timer2 = timer("net forwarding")
        self.timer3 = timer("back propagation")
        self.timer4 = timer("evaluation")
        self.timer5 = timer("writing files")

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

        # model.load()
        model.net = torch.nn.DataParallel(model.net)
        model.cuda()

        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("--------------------------\n "
                             "      [*] Using AMP !\n"
                             " -------------------------")

        return model, trainloader, valloader

    def fit(self, model, trainset, valset=None,  trainloader=None, valloader=None, testloader=None):
        model, trainloader, valloader = self.init_model(model, trainset)

        # Set optimizer and scheduler
        optim_configs = model.configure_optimizers()
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        for epoch in range(self.max_epochs):
            self.on_before_zero_grad()
            # Training
            self.train(model, trainloader, epoch, optimizer, scheduler)
            # Evaluation
            if epoch % self.val_seq == 0:
                if valset is not None:
                    self.validate(model, valset, epoch)
                if self.tester is not None:
                    out = self.tester.test(model, epoch)
                    if self.monitor is not None:
                        best_dict = self.monitor.record(out, epoch)
                        self.recorder_test.record({**best_dict, **out})
                        if best_dict['isbest']:
                            self.save_best_model(model, epoch)
                        self.logger.info(f"[*] {dict_to_str(best_dict)} || Epoch {epoch}: {dict_to_str(out)}")
                        self.csvlogger.record({**best_dict, **out, "time":_get_time_str()})
                    else:
                        self.logger.info(f"[*] Epoch {epoch}: {dict_to_str(out)}")
            self.autosave(model, epoch)

    def on_before_zero_grad(self, **kwargs):
        pass

    def train(self, model, trainloader, epoch, optimizer, scheduler=None):
        model.train()
        self.recorder.clear()
        for batch_idx, data in tqdm(enumerate(trainloader), ncols=100):
            optimizer.zero_grad()
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.cuda()
            if self.use_amp:
                with autocast():
                    out = model.training_step(data, batch_idx)
                    loss = out['loss']
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                out = model.training_step(data, batch_idx)
                loss = out['loss']
                loss.backward()
                optimizer.step()
            self.recorder.record(out)
            if epoch == 0:
                self.logger.info("[*] Debug Checking Pipeline !!!")
                break
            # print()
        if scheduler is not None:
            scheduler.step()
        loss_values, loss_keys = self.recorder.cal_metrics()
        loss_str = ""
        for i, v in enumerate(loss_keys):
            loss_str += "{}:{:.3f} ".format(v, loss_values[i]) # f"{v}:{loss_values[i]}; "
        self.logger.info(f"Epoch {epoch}: {loss_str}")

    def validate(self, model, valloader, epoch):
        model.eval()
        with torch.no_grad():
            self.recorder.clear()
            for batch_idx, data in enumerate(valloader):
                for k, v in data.items():
                    if type(v) == torch.Tensor:
                        data[k] = v.cuda()
                out = model.validation_step(data, batch_idx)
                self.recorder.record(out)
            loss_values, loss_keys = self.recorder.cal_metrics()
            loss_str = ""
            for i, v in enumerate(loss_keys):
                loss_str += f"{v}:{loss_values[i]}; "
            print(f"Validation step, Epoch {epoch}: {loss_str}")
        return loss_values

    def _init_test(self, model, testset):
        testloader = DataLoader(dataset=testset,
                               batch_size=1, num_workers=1)
        model.load_ckpt()
        model.net = torch.nn.DataParallel(model.net)
        model.cuda()
        return model, testloader

    def test(self, model, testset):
        model, testloader = self._init_test(model, testset)
        model.eval()
        with torch.no_grad():
            self.recorder.clear()
            for batch_idx, data in enumerate(testloader):
                for k, v in data.items():
                    if type(v) == torch.Tensor:
                        data[k] = v.cuda()
                out = model.testing_step(data, batch_idx)
                self.recorder.record(out)
            # After Inference
            loss_values, loss_keys = self.recorder.cal_metrics()
            loss_str = ""
            for i, v in enumerate(loss_keys):
                loss_str += f"{v}:{loss_values[i]}; "
            self.logger.info(f"Testing: {loss_str}")
            info_dict = {"tag": self.tag}
        return loss_values

    def autosave(self, model, epoch):
        if self.save_latest_only:
            return self.save_latest(model, epoch)
        if self.save_seq > 0 and epoch % self.save_seq == 0:
            return self.save(model, epoch)

    def save_latest(self, model, epoch):        
        torch.save(model.net.state_dict(), self.runs_dir + "/model_latest.pth".format(epoch))
        self.logger.info(f"Epoch {epoch}: Just Saved model to ``{self.runs_dir + '/model_latest.pth'.format(epoch)}``! ")

    def save(self, model, epoch):
        torch.save(model.net.state_dict(), self.runs_dir + "/model_epoch_{}.pth".format(epoch))
        self.logger.info(f"Epoch {epoch}: Just Saved model to ``{self.runs_dir + '/model_epoch_{}.pth'.format(epoch)}``! ")

    def save_best_model(self, model, epoch):        
        torch.save(model.net.state_dict(), self.runs_dir + "/best_model_epoch_{}.pth".format(epoch))
        self.logger.info(f"[Best model] Epoch {epoch}: Saved best model to ``{self.runs_dir + '/model_epoch_{}.pth'.format(epoch)}``! ")

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