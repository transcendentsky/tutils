import time
import torch
import torchvision
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset as _Dataset
from .tutils import tfilename


class tenum:
    def __init__(self, iter):
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
        return (end-start), i, res
      

def tfunctime(func):
    def run(*argv, **kargs):
        t1 = time.time()
        ret = func(*argv, **kargs)
        t2 = time.time()
        # print(f"[Function {func.__name__}] Running time:{(t2-t1):.6f}s")
        return (t2-t1), ret
    return run


# print(isinstance(g, Iterable)) # true
# print(isinstance(g, Iterator)) # true
# print(isinstance(g, Generator)) # false