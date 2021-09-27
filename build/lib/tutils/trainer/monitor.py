


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
