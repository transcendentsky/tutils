import SimpleITK as sitk
import numpy as np
import os
import sys
import cv2

# ----------------------
from medpy.io import load

def read2(*args, **kwargs):
    load(*args, **kwargs)


type_dict = {"nifti": "NiftiImageIO",
             "nii"  : "NiftiImageIO",  
             "nrrd" : "NrrdImageIO" ,
             "jpg"  : "JPEGImageIO" ,
             "jpeg" : "JPEGImageIO" , 
             "png"  : "PNGImageIO"  ,
             }

def read(path, mode:str="nifti"):
    mode = mode.lower()
    if mode in type_dict.keys():
        reader = sitk.ImageFileReader()
        reader.SetImageIO(type_dict[mode])
        reader.SetFileName(path)
        image = reader.Execute()
        
        return image
    elif mode in ['dicom', 'dcm']:
        assert os.path.isdir(path)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        dicom_image = reader.Execute()
        # dicom_np = sitk.GetArrayFromImage(dicom_image) 
        return dicom_image
    else:
        raise NotImplementedError()