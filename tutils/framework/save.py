import torch

class Tester(object):
    def __init__(self, logger, config, dataset):
        self.loss_total = 0
        self.loss_list = []
        self.logger = logger
        self.config = config

    def reset(self):
        self.loss_list.clear()
        self.loss_total = 0

    def record(self, loss):
        self.loss_list.append(loss)

    def test(self, net, ):
        pass

def save_model(logger, config):
    # state = model.state_dict()
    torch.save(model.state_dict(), tfilename(config['runs_dir'], "model", "best_{}.pkl".format(epoch)))