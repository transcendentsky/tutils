# coding: utf-8
import numpy as np
import math
import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import cv2
from ._eval_ones import cal_CC, cal_PSNR, cal_MSE, cal_MAE  # cal_SSIM
import torch
from ..tutils import *

def np2tensor(batch_a):
    a = batch_a.transpose((0,3,1,2))
    tensor_a = torch.from_numpy(a)
    return tensor_a

def tensor2np(batch_a):
    a = batch_a.detach().cpu().numpy().transpose((0, 2, 3, 1))
    return a

def cal_metrix_batch(img1, img2, metrix):
    if type(img1) == torch.Tensor:
        img1 = tensor2np(img1)
    if type(img2) == torch.Tensor:
        img2 = tensor2np(img2)
    return cal_metrix_np_batch

def cal_metrix_tensor_batch(img1_tensor, img2_tensor, metrix):
    img1 = tensor2np(img1_tensor)
    img2 = tensor2np(img2_tensor)
    return cal_metrix_np_batch(img1, img2, metrix)

# def cal_ssim_tensor_batch(img1, img2):
#     # # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
#     # # Y: (N,3,H,W)  
#     ssim_loss = pytorch_ssim.SSIM(window_size = 11)
#     return ssim_loss(img1, img2)

def cal_metrix_np_batch(img1, img2, metrix):
    """
    Cal all metrix with two Numpy matrix
    """
    # assert np.max(img1) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img1))
    # assert np.max(img2) > 1, "The value should be [0, 255], max_value is {}".format(np.max(img2))
    assert np.ndim(img1) == 4, "np.ndim Error ! Got {}".format(img1.shape)
    assert np.ndim(img2) == 4, "np.ndim Error ! Got {}".format(img2.shape)

    criterion_dict = {
        "cc": cal_CC, 
        "psnr": cal_PSNR,
        "ssim": cal_SSIM,
        "mse": cal_MSE,
        "mae": cal_MAE 
    }
    criterion = criterion_dict[metrix]

    bs = img1.shape[0]
    loss = 0.0
    for i in range(bs):
        loss += criterion(img1[i, :, :, :], img2[i, :, :, :])
        # print(loss)
    return loss/(bs*1.0), loss

if __name__ == "__main__":
    a = np.random.rand(300,300,3)
    b = np.random.rand(300,300,3)
    aa = cv2.imread("medical/corgi1.jpg")
    bb = aa
    print(cal_SSIM(a, b, rgb=True))