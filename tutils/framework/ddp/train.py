# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler


if __name__ == "__main__":
    model = None
    torch.distributed.init_process_group(backend="nccl")
    model=torch.nn.parallel.DistributedDataParallel(model)

    # 需要注意的是：DDP并不会自动shard数据 1. 如果自己写数据流，得根据torch.distributed.get_rank()去shard数据，
    # 获取自己应用的一份 2. 如果用Dataset API，则需要在定义Dataloader的时候用DistributedSampler 去shard：
    sampler = DistributedSampler(dataset) # 这个sampler会自动分配数据到各个gpu上
    DataLoader(dataset, batch_size=batch_size, sampler=sampler)