import time
import torch
import torchvision
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset as _Dataset
from .tutils import tfilename
import numpy as np
import datetime


def trans_time():
    """
        print time str of now
    """
    return datetime.datetime.now().strftime('%m%d-%H%M%S')


class tenum:
    def __init__(self, iter):
        self.stop_time = time.time()
        if type(iter) != enumerate:
            self.iter = enumerate(iter)
        else:
            self.iter = iter

    def __iter__(self):
        return self
    
    def __next__(self):
        start = time.time()
        i, res = self.iter.__next__()
        end = time.time()
        iter_time = end - self.stop_time
        self.stop_time = end
        return (end-start), i, res


def format_result(result):
    date = datetime.datetime.utcfromtimestamp(result)
    output = datetime.datetime.strftime(date, "%M:%S:%f")
    return output


class timer(object):
    def __init__(self, flag=None, verbose=False):
        self.flag = flag
        self.verbose = verbose
        self.time_list = []
        self.start_time = time.time()
        
    def __call__(self):
        return self.stop()
        
    def start(self):
        self.start_time = time.time()
    
    def one_stop(self, verbose=False):
        self.stop_time = time.time()
        interval = self.stop_time - self.start_time
        self.start_time = self.stop_time
        self.time_list.append(interval)
        if verbose:
            print(f"interval: {interval}")
        return interval
    
    def stop(self):
        interval = time.time() - self.start_time
        self.start_time = time.time()
        return interval
        
    def clear(self):
        self.time_list.clear()

    def sum(self):
        _list = np.array(self.time_list)
        return np.sum(_list)

    def avg(self):
        _list = np.array(self.time_list)
        return np.mean(_list)
        
    def showavg(self):
        if self.flag is not None:
            return f"{self.flag}:{format_result(self.avg())}"
        else:
            return format_result(self.avg())
        
    def showsum(self):
        if self.flag is not None:
            return f"{self.flag}:{format_result(self.sum())}"
        else:
            return format_result(self.sum())
        
def tfunctime(func):
    def run(*argv, **kargs):
        t1 = time.time()
        ret = func(*argv, **kargs)
        t2 = time.time()
        # print(f"[Function {func.__name__}] Running time:{(t2-t1):.6f}s")
        return (t2-t1), ret
    return run


def tshowtime(func):
    def run(*argv, **kargs):
        t1 = time.time()
        ret = func(*argv, **kargs)
        t2 = time.time()
        print(f"[Function {func.__name__}] Running time:{(t2-t1):.6f}s")
        return ret
    return run

# print(isinstance(g, Iterable)) # true
# print(isinstance(g, Iterator)) # true
# print(isinstance(g, Generator)) # false
if __name__ == "__main__":
    t1 = timer("load_time")
    t2 = timer("test_time")
    t1()
    t2()
    time.sleep(500)
    t1.stop()
    t2.stop()
    print(t1.showavg())
    print(t2.showsum())
