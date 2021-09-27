"""
    This script is a Demo !
"""

import torch
import torchvision
import monai
import numpy as np


aa = np.arange(3*5*5).reshape((1,3,5,5))
bb = np.arange(3*5*5).reshape((1,3,5,5))
adict = {'0':aa, '1':bb}


def usage(adict=adict):
    # Rotate
    trans = monai.transforms.RandRotated(keys=['0', '1'], )
    # Crop    
    trans = monai.transforms.RandSpatialCropd(keys=['0', '1'], random_size=False, roi_size=(2,2,2))
    # 

    res = trans(adict)
    return res