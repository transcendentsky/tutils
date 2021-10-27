

class MultiOptimizer(object):
    def __init__(self, optimizer_list):
        self.optimizer_list = optimizer_list

    def step(self):
        for optimizer in self.optimizer_list:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizer_list:
            optimizer.zero_grad()

    def get_lr(self):
        return self.optimizer_list[0].param_groups[0]['lr']


class MultiScheduler(object):
    def __init__(self, sche_list):
        self.sche_list = sche_list

    def step(self):
        for sche in self.sche_list:
            sche.step()

    def zero_grad(self):
        for sche in self.sche_list:
            sche.zero_grad()

    def get_lr(self):
        return self.sche_list[0].optimizer.param_groups[0]['lr']

