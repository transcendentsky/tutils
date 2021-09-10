import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader

def check_nan(data):
    data = np.array(data)
    a = np.isnan(np.sum(data))
    return a  


class BaseDataset(Dataset):
    def __init__(self, image_dir):
        super(BaseDataset, self).__init__()
        self.image_dir = image_dir
        
    def __getitem__(self, index):
        pass
    
    