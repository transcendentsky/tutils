import torch
from torch import nn
from collections import OrderedDict


class LearnerModule(nn.Module):
    def __init__(self, config=None, logger=None, **kwargs):
        super(LearnerModule, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, x, **kwargs):
        # return self.net(x)
        pass

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

    def on_before_zero_grad(self, **kwargs):
        pass

    def configure_optimizers(self, **kwargs):
        # optimizer = optim.Adam(params=self.parameters(), \
        #                    lr=self.config['optim']['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
        #                    weight_decay=self.config['optim']['weight_decay'])
        # return {'optimizer': optimizer, "scheduler": None}
        pass

    def validation_step(self, data, batch_idx, **kwargs):
        pass

    def testing_step(self, data, batch_idx, **kwargs):
        pass

    def load(self):
        pass
        # assert os.path.exists(ckpt), f"{ckpt}"
        # print(f'Load CKPT {ckpt}')
        # state_dict = torch.load(ckpt)
        # self.net.load_state_dict(state_dict)

    # def load_state_dict(self, state_dict):
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         if not k.startswith("module.target_encoder.net."):
    #             print("skip: ", k)
    #             continue
    #         name = k.replace("module.target_encoder.net.", "")
    #         new_state_dict[name] = v
    #     self.net.load_state_dict(new_state_dict)
