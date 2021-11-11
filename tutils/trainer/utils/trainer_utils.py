from datetime import datetime



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


class VoidTimer(object):
    def __init__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        return None


def dict_to_str(d):
    loss_str = ""
    for k, v in d.items():
        loss_str += "\t {}\t: {} \n".format(k, v)  # f"{v}:{loss_values[i]}; "
    return loss_str


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')