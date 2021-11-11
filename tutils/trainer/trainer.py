# coding: utf-8
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from .trainer_abstract import AbstractTrainer

# def trainer_from_config(logger, config, tester, monitor, **kwargs):
#     config_base = config['base']
#     config_train = config['training']
#     return Trainer(logger=logger,
#                    config=config,
#                    tester=tester,
#                    monitor=monitor,
#                    mode="ps",
#                    runs_dir=config_base['runs_dir'],
#                    tag=config_base['tag'],
#                    num_epochs=config_train['num_epochs'],
#                    batch_size=config_train['batch_size'],
#                    num_workers=config_train.get('num_workers', 0),
#                    save_interval=config_train.get('save_interval', 50),
#                    use_amp=config_train.get('use_amp', False),
#                    val_check_interval=config_train.get('val_check_interval', 50),
#                    save_latest_only=config_train.get('save_latest_only', False),
#                    training_log_interval=config_train.get('training_log_interval', 1),
#                    load_pretrain_model=config_train.get('load_pretrain_model', False),
#                    gpus=config_train.get('gpus', 4),
#                    kwargs=kwargs,
#                    )


class Trainer(AbstractTrainer):
    def __init__(self, 
                logger=None, 
                config=None,
                tester=None, 
                monitor=None,
                **kwargs):
        """
            mode:
                ps:  Parameter Server
                ddp: Distributed data parallel
        """
        super(Trainer, self).__init__(config, tester, monitor, rank='cuda', world_size=0)
        self.logger = logger


    def init_model(self, model, trainset, **kwargs):
        assert len(trainset) > 0 , f"Got {len(trainset)}"
        self.trainloader = DataLoader(dataset=trainset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True)
        if self.load_pretrain_model:
            model.load()
        model.net = torch.nn.DataParallel(model.net)
        model.cuda()

        return model


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