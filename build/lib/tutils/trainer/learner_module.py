import torch
from torch import nn
from collections import OrderedDict
from tutils import MultiLogger
import os
from abc import ABC, abstractmethod


class LearnerModule(nn.Module):
    def __init__(self, config=None, logger=None, **kwargs):
        super(LearnerModule, self).__init__()
        self.config = config
        self.logger = logger
        self.net = None

    @abstractmethod
    def forward(self, x, **kwargs):
        # return self.net(x)
        pass

    @abstractmethod
    def training_step(self, data, batch_idx, **kwargs):
        # img1 = data["crp_1"]
        # img2 = data["crp_2"]
        # p1 = data['point_crp_1']
        # p2 = data['point_crp_2']

        # y1_list, y2_list = self.forward(img1, img2)

        # loss = loss_1 + loss_2 + loss_3 + loss_4

        # return {'loss': loss, "loss1": loss_1, "loss2": loss_2, "loss3": loss_3, "loss4": loss_4,
        #         "sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4}
        pass

    @abstractmethod
    def configure_optimizers(self, **kwargs):
        # optimizer = optim.Adam(params=self.parameters(), \
        #                    lr=self.config['optim']['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
        #                    weight_decay=self.config['optim']['weight_decay'])
        # return {'optimizer': optimizer, "scheduler": None}
        pass

    @abstractmethod
    def validation_step(self, data, batch_idx, **kwargs):
        pass

    @abstractmethod
    def testing_step(self, data, batch_idx, **kwargs):
        pass

    def on_before_zero_grad(self, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        if self.logger is not None:
            self.logger.info(msg, *args, **kwargs)
        else:
            print(msg, *args, **kwargs)

    def load(self, pth=None, *args, **kwargs):
        assert os.path.exists(pth)
        print(f'Load CKPT from ``{pth}``')
        state_dict = torch.load(pth)
        self.net.load_state_dict(state_dict)

        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if not k.startswith("module.target_encoder.net."):
        #         print("skip: ", k)
        #         continue
        #     name = k.replace("module.target_encoder.net.", "")
        #     new_state_dict[name] = v
        # self.net.load_state_dict(new_state_dict)

    def save(self, pth, *args, **kwargs):
        # Default: "/model_epoch_{}.pth".format(epoch)
        torch.save(self.net.module.state_dict(), pth)
        return True

    def configure_logger(self, *args, **kwargs):
        logger = MultiLogger(logdir=self.config['base']['runs_dir'],
                             mode=self.config['logger']['mode'],
                             tag=self.config['base']['tag'],
                             extag=self.config['base'].get('experiment', None),
                             action=self.config['logger'].get('action', 'k'))
        return {'logger': logger}

    def save_optim(self, pth, optimizer, epoch=0, *args, **kwargs):
        stat = {'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(stat, pth)
        return True
        # pass

    def load_optim(self, optimizer, pth=None, *args):
        # state_dict = torch.load(pth)
        # optimizer.load_state_dict(state_dict['optimizer'])
        # start_epoch = state_dict.get('epoch', 0) + 1
        # return start_epoch
        return 0