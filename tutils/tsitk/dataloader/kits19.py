# coding: utf-8

import os
import numpy as np
import torch
import random
import torchvision
import string

import random
import time
import cv2
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from tqdm import tqdm


## Dataset pipeline
#  Resample to the same spacing
#  cut off to [-79, 304] , and Z-Score
#  remove some bad instances and correct some wrong labels
#
# Dataset Structure
# /path/to/kits19/data/
#     case_00000/
#         imaging.nii.gz
#         segmentation.nii.gz
#     case_00xxx
#     ......

class Kits19(Dataset):
    
    def __init__(self, load_mod:str="all", datadir:str="/home1/quanquan/datasets/kits19/data"):
        self.datadir  = datadir
        self.load_mod = load_mod
        
        self.dirnames = np.array([x.name for x in os.scandir(datadir) if (os.path.isdir(x.path) and x.name.startswith("case_"))])
        self.dirnames.sort()
        
    def __getitem__(self, index):
        image_name = "imaging.nii.gz"
        label_name = "segmentation.nii.gz"
        
        image_path = os.path.join(self.datadir, self.dirnames[index], image_name)
        label_path = os.path.join(self.datadir, self.dirnames[index], label_name)
        
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(image_path)
        image = reader.Execute()
        
        if self.load_mod == "img_only":
            # check if clipped:
            img_np = sitk.GetArrayFromImage(image)
            assert np.max(img_np) <= 304 and np.min(img_np) >= -79
            return img_np
        
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(label_path)
        label = reader.Execute()
        
        if self.load_mod == "resample_kits":
            # resample to 3.22*1.62*1.62
            new_spacing = (3.22,1.62,1.62)
            print("resampled data to ", new_spacing)
            image = resampleImage(image, NewSpacing=new_spacing)
            label = resampleImage(label, NewSpacing=new_spacing)

            # clipping 
            new_image = sitk.Clamp(image, lowerBound=-79, upperBound=304)
            new_label = sitk.Clamp(label, lowerBound=-79, upperBound=304)
            return new_image, new_label
        
        if self.load_mod == "all":
            return image, label
        
        if self.load_mod == "np":
            return sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(label)
        
    def z_score(self, index, output_dir, avg, delta):
        image, label = self.__getitem__(index)
        image_scored = (image*1.0 - avg)/delta
        label_scored = (label*1.0 - avg)/delta
        
        image = sitk.GetImageFromArray(image_scored)
        label = sitk.GetImageFromArray(label_scored)
        
        output_image_path = tfilename(output_dir, self.dirnames[index], "imaging.nii.gz")
        output_label_path = tfilename(output_dir, self.dirnames[index], "segmentation.nii.gz")
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_image_path)
        writer.Execute(image)
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_label_path)
        writer.Execute(label)       
        # print("Written in {} and its label".format(output_image_path))
        
    def z_score_dataset(self, output_dir, avg, delta):
        print("Starting Z-socre dataset from ", self.datadir, "To", output_dir)
        self.load_mod = "np"
        for index in tqdm(range(self.__len__())):
            self.z_score(index, output_dir, avg, delta)
    
    def resample_data(self, index, output_dir):
        new_image, new_label = self.__getitem__(index)
        
        output_image_path = tfilename(output_dir, self.dirnames[index], "imaging.nii.gz")
        output_label_path = tfilename(output_dir, self.dirnames[index], "segmentation.nii.gz")
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_image_path)
        writer.Execute(new_image)
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_label_path)
        writer.Execute(new_label)       
        print("Written in {} and its label".format(output_image_path))

    def resample_dataset(self, output_dir):
        print("Starting Resampling dataset from ", self.datadir, "To", output_dir)
        self.load_mod = "resample_kits"
        for index in range(self.__len__()):
            self.resample_data(index, output_dir)
        
    def __len__(self):
        return len(self.dirnames)



# -------------  Z-score  ----------------
def _z_score_1():
    # First: get the avg.
    avg_total = 0
   
    kits = Kits19(load_mod="img_only", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    _len = len(kits)
    for i in tqdm(range(_len)):
        img_np = kits.__getitem__(i)
        avg_total += np.mean(img_np)
    avg = avg_total * 1.0 / _len 
    print(avg)
    np.save("kits-avg.npy", avg)

def _z_score_2():
    # Second: get the delta / std
    delta_total = 0
    dim_total = 0
    avg = np.load("kits-avg.npy")
    
    kits = Kits19(load_mod="img_only", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    _len = len(kits)
    for i in tqdm(range(_len)):
        img_np = kits.__getitem__(i)
        # h,w,c = img_np.shape
        # dim = h*w*c
        img_flat = img_np.flatten()
        arr_len = img_flat.shape[0]
        dim_total += arr_len
        for v in img_flat:
            delta_total += (v - avg)**2

    delta = np.sqrt(delta_total*1.0 / dim_total)
    print("delta:", delta)
    np.save("kits-delta.npy", delta)

def _z_score_3():
    # Third: normalize data
    avg = np.load("kits-avg.npy")
    print("Avg: ", avg)
    delta = np.load("kits-delta.npy")
    print("Detal: ", delta)
    # exit(0)
    kits = Kits19(load_mod="np", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    kits.z_score_dataset("/home1/quanquan/datasets/kits19/normaled_data", avg, delta)
        

# -----------------------  Test  -----------------------------
def test_z_score_0():
    kits = Kits19(load_mod="np", datadir="/home1/quanquan/datasets/kits19/resampled_data")
    _len = len(kits)
    image, label = kits.__getitem__(5)
    avg = np.mean(image)
    std = np.std(image)
    print(avg, std)