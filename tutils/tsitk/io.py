import SimpleITK as sitk
import numpy as np
import os
import sys
import cv2

# ----------------------
# A useful package for loading radiological images
from medpy.io import load

def read2(*args, **kwargs):
    image_data, image_header = load(*args, **kwargs)
    return image_data, image_header


type_dict = {"nifti": "NiftiImageIO",
             "nii"  : "NiftiImageIO",  
             "nrrd" : "NrrdImageIO" ,
             "jpg"  : "JPEGImageIO" ,
             "jpeg" : "JPEGImageIO" , 
             "png"  : "PNGImageIO"  ,
             }

def read(path:str, mode:str="nifti") -> sitk.SimpleITK.Image:
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
    
def write(img:sitk.SimpleITK.Image, path:str, mode:str="nifti"):
    """
    Path: (example) os.path.join(jpg_dir, f"trans_{random_name}.nii.gz")
    """
    mode = mode.lower()
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(img)

def np_to_itk(img_np) -> np.ndarray:
    return sitk.GetImageFromArray(img_np)

def itk_to_np(img) -> sitk.SimpleITK.Image:
    return sitk.GetArrayFromImage(img)

