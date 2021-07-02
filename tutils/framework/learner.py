import torch
from torch import nn
from collections import OrderedDict


class LearnerModule(nn.Module):
    def __init__(self, config=None, logger=None, **kwargs):
        super(LearnerModule, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, x, **kwargs):
        pass

    def training_step(self, data, batch_idx, **kwargs):
        pass

    def on_before_zero_grad(self, **kwargs):
        pass

    def configure_optimizers(self, **kwargs):
        pass

    def validation_step(self):
        pass

    def load(self, ckpt):
        assert os.path.exists(ckpt), f"{ckpt}"
        print(f'Load CKPT {ckpt}')
        state_dict = torch.load(ckpt)
        self.net.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith("module.target_encoder.net."):
                print("skip: ", k)
                continue
            name = k.replace("module.target_encoder.net.", "")
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
