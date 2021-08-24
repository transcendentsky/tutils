"""
    Noisy Students,
"""

import torch
import torchvision
from tutils import tfilename, trans_args, trans_init, dump_yaml, tdir, save_script
from tutils.framework import Trainer, LearnerModule, Monitor
import numpy as np
from networks.loss import focal_loss, L1Loss
from torch import optim
from torch.optim.lr_scheduler import StepLR
import argparse
from networks.network_emb_study import UNet_Pretrained2
from utils.tester import Tester
from datasets.data_loader import Cephalometric
from datasets.pseudo_dataset import PseudoDataset


class Learner(LearnerModule):
    def __init__(self, config, logger):
        super(Learner, self).__init__(config, logger)
        self.net = UNet_Pretrained2(3, config['special']['num_landmarks'])
        self.loss_logic_fn = focal_loss
        self.loss_regression_fn = L1Loss
        self.lbda = config['special']['lambda']

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        img, mask, offset_y, offset_x, landmark_list = data['img'], data['mask'], data['offset_y'], data['offset_x'], data['landmark_list']

        heatmap, regression_y, regression_x = self.forward(img)

        logic_loss = self.loss_logic_fn(heatmap, mask)
        regression_loss_y = self.loss_regression_fn(regression_y, offset_y, mask)
        regression_loss_x = self.loss_regression_fn(regression_x, offset_x, mask)

        loss = regression_loss_x + regression_loss_y + logic_loss * self.lbda
        # print("debug", loss)
        return {"loss": loss, "regress_loss_x": regression_loss_x, "regression_loss_y": regression_loss_y, "logic_loss": logic_loss}

    def configure_optimizers(self, **kwargs):
        optimizer = optim.Adam(params=self.net.parameters(), lr=self.config['optim']['learning_rate'], betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=self.config['optim']['weight_decay'])
        scheduler = StepLR(optimizer, self.config['optim']['decay_step'], gamma=self.config['optim']['decay_gamma'])
        return {'optimizer': optimizer, "scheduler": scheduler}

    def load(self, path=None):
        ckpt_path = self.config['network']['ckpt'] if path is None else path
        self.logger.info(f"Load Pretrain model `{ckpt_path}`")
        state_dict = torch.load(ckpt_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            # name = k.replace("module.", "")
            name = k[7:]
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)


def check(data, ids):
    # assert len(ids) == 2
    rlist = []
    # data_reshape = rearrange(data, "n ")
    for i in range(150):
        one = data[ids, i, :]  # shape [2, 287]
        conf_idx = np.argsort(one, axis=0)
        r = np.take_along_axis(one, conf_idx, axis=0)[-1]
        rlist.append(r)
    return np.array(rlist)


def train(logger, config, args):
    num_ref = config['num_ref']

    # Moniter: key is the key in `Tester` to be detected
    monitor = Monitor(key='mre', mode='dec')
    # Tester: tester.test should return {"mre": mre, ...}
    tester_subtest = Tester(logger, config, mode="subtest")

    model = Learner(config, logger)
    logger.info(f"Training with LIMITED samples")
    dataset = Cephalometric(config['dataset']['pth'], random_idx=[1,2,3,4])

    ######################  Trainer Settings  ###############################
    trainer = Trainer(logger, config, tester=tester_subtest, monitor=monitor,
                      tag=config['tag'], runs_dir=config['runs_dir'], val_seq=config['validation']['val_seq'], **config['training'])



    if args.pretrain:
        model.load()
        model.cuda()
    else:
        trainer.fit(model, dataset)

def test(logger, config, args):
    model = Learner(config, logger)
    epoch = args.epoch
    pth = tfilename(config['runs_dir'], f"model_epoch_{epoch}.pth")
    model.load(pth)
    model.cuda()
    tester_train = Tester(logger, config, mode="Train")
    tester_test = Tester(logger, config, mode="Test1+2")
    logger.info(f"Dataset Training")
    tester_train.test(model)
    logger.info(f"Dataset Test 1+2")
    tester_test.test(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/random.yaml")
    parser.add_argument("--num_ref", type=int, default=None)
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--nodump", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epoch", type=int, default=800)
    # parser.add_argument("--random_idx", action="store_true")
    parser.add_argument("--select_idx", action="store_true")
    args = trans_args(parser)
    logger, config = trans_init(args)
    save_script(config['runs_dir'], __file__)
    if args.test:
        test(logger, config, args)
    else:
        train(logger, config, args)
