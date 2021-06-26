# coding: utf-8
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
# trans utils
from tutils import timer



class Trainer(object):
    def __init__(self, logger=None, config=None, mode="ps", gpus=4, sync_batchnorm=True):
        """
            mode:
                ps:  Parameter Server
                ddp: Distributed data parallel
        """
        assert mode in ['ps', "ddp"]
        self.mode = mode
        if mode == "ddp": self.init_ddp_env()
        self.logger = logger
        self.init_timers()
        # Config
        self.config = config
        self.max_epochs = config['training']['num_epochs']
        self.batch_size = config["training"]['batch_size']
        self.num_workers = config['training']['num_workers']
        self.save_seq = config['training']['save_seq']
        self.val_seq = config['validation']['val_seq']

        # Other
        self.recorder = Recoder()
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def init_timers(self):
        self.timer1 = timer("data loading")
        self.timer2 = timer("net forwarding")
        self.timer3 = timer("back propagation")
        self.timer4 = timer("evaluation")
        self.timer5 = timer("writing files")

    def init_model(self, model, trainset):
        if self.mode == "ddp":
            return self.init_ddp(model, trainset)

    def init_ps(self, model, trainset):
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=self.batch_size,num_workers=self.num_workers)
        model.to(self.device)
        return model, trainloader

    def fit(self, model:LearnerModule, trainset, valset=None,  trainloader=None, valloader=None, testloader=None):
        model, trainloader = self.init_model(model, trainset)

        optim_configs = model.configure_optimizers()
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        for epoch in range(self.max_epochs):
            self.recorder.clear()
            for batch_idx, data in enumerate(trainloader):
                if torch.cuda.is_available():
                    data = data.to(self.device)

                model.on_before_zero_grad()
                optimizer.zero_grad()
                out = model.training_step(data, batch_idx)
                loss = out['loss']
                loss.backward()
                optimizer.step()
                self.recorder.record(loss)

            scheduler.step()
            print(f"Epoch {epoch}: loss {self.recorder.cal_metrics()}")

            if valset is not None:
                pass

            if self.save_seq > 0 and epoch % self.save_seq == 0:
                torch.save(model.state_dict(), self.config['runs_dir'] + "/model_epoch_{}.pth".format(epoch))
                print(f"Epoch {epoch}: Just Saving models! ")

    def test(self, model, testset):
        pass

    def init_ddp_env(self):
        # 1 Initialize
        torch.distributed.init_process_group(backend="nccl")
        # 2 set up gpu for each process
        self.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        print("debug: ", self.local_rank, self.device)

    def init_ddp(self, model, trainset):
        # 3 Use DistributedSampler
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=self.batch_size,
                                 sampler=DistributedSampler(trainset))
        # 4 move the network to the predetermined gpu
        model.to(self.device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # 5) pack up with DistributedDataParallel
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[self.local_rank],
                                                              output_device=self.local_rank)
        return model, trainloader

    def cleanup(self):
        torch.distributed.destroy_process_group()



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