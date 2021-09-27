"""
    A developmental version of radiological features
"""
import numpy as np
import SimpleITK as sitk
import six

from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, gldm
from typing import List, Dict, NoReturn

def calc_radio_fea(img:np.ndarray, mask:np.ndarray) -> List[np.ndarray]:
    assert type(img) == np.ndarray, f"TypeError, expected np.ndarray but Got {type(img)}"
    assert img.shape == mask.shape, f"SizeError, expected to be same, but Got {img.shape} and {mask.shape}"
    
    image = sitk.GetImageFromArray(img)
    mask  = sitk.GetImageFromArray(mask)

    # Setting for the feature calculation.
    # Currently, resampling is disabled.
    # Can be enabled by setting 'resampledPixelSpacing' to a list of 3 floats (new voxel size in mm for x, y and z)
    settings = {'binWidth': 25,
                'interpolator': sitk.sitkBSpline,
                'resampledPixelSpacing': None}
    #
    # If enabled, resample image (resampled image is automatically cropped.
    #
    interpolator = settings.get('interpolator')
    resampledPixelSpacing = settings.get('resampledPixelSpacing')
    if interpolator is not None and resampledPixelSpacing is not None:
        image, mask = imageoperations.resampleImage(image, mask, **settings)

    bb, correctedMask = imageoperations.checkMask(image, mask)
    if correctedMask is not None:
        mask = correctedMask
    image, mask = imageoperations.cropToTumorMask(image, mask, bb)
    results_collect = dict()
    results_np = list()
    # Fisrt order
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
    # firstOrderFeatures.enableFeatureByName('Mean', True)
    firstOrderFeatures.enableAllFeatures()
    results:dict = firstOrderFeatures.execute() # dict()
    # results_collect['FirstOrder'] = results
    results_np.append(np.array([value for key, value in results.items()]))
    # 
    shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
    shapeFeatures.enableAllFeatures()
    results = shapeFeatures.execute()
    # results_collect['ShapeFeature'] = results
    results_np.append(np.array([value for key, value in results.items()]))
    ###
    glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    # results_collect['GLCM'] = results
    results_np.append(np.array([value for key, value in results.items()]))
    ###
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
    glrlmFeatures.enableAllFeatures()
    results = glrlmFeatures.execute()
    # results_collect['GLRLM'] = results
    results_np.append(np.array([value for key, value in results.items()]))
    ###
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
    glszmFeatures.enableAllFeatures()
    results = glszmFeatures.execute()
    # results_collect['GLSZM'] = results
    results_np.append(np.array([value for key, value in results.items()]))
    
    gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
    gldmFeatures.enableAllFeatures()
    results = gldmFeatures.execute()
    results_np.append(np.array([value for key, value in results.items()]))
    
    return results_np


def compare_radio_feas(fea1:list, fea2:list):
    e = 1e-8
    loss_list = list()
    var1_list = list()
    var2_list = list()
    for ttype in range(len(fea1)):
        total_loss = list()
        var1 = list()
        var2 = list()
        for key in range(len(fea1[ttype])):
            value1 = np.float(fea1[ttype][key])
            value2 = np.float(fea2[ttype][key])
            loss = np.sqrt((value1-value2)**2.0)
            total_loss.append(loss)
            var1.append(loss/(value1+e))
            var2.append(loss/(value2+e))
        loss_list.append(np.array(total_loss))
        var1_list.append(np.array(var1))
        var2_list.append(np.array(var2))
    return loss_list, var1_list, var2_list

def compare_radio_feas_from_imgs(image1:np.ndarray, image2:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
    Compare the radiomics features bwteen two imgs
    """
    return compare_radio_feas_info(image1, image2, mask)

def compare_radio_feas_info(image1:np.ndarray, image2:np.ndarray, mask:np.ndarray) -> np.ndarray:
    fea1 = calc_radio_fea(image1, mask)
    fea2 = calc_radio_fea(image2, mask)
    loss_list, var1_list, var2_list = compare_radio_feas(fea1, fea2)
    results = []
    for var2 in var2_list:
        avg_var = np.mean(var2)
        results.append(avg_var)
    return np.array(results)

def usage() -> NoReturn:
    imageName, maskName = getTestCase('brain1')
    image = sitk.ReadImage(imageName)
    
    mask = sitk.ReadImage(maskName)
    img_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)

    image1 = img_np[13,:,:][:,:,np.newaxis]
    image2 = img_np[14,:,:][:,:,np.newaxis]
    mask1  = mask_np[13,:,:][:,:,np.newaxis]
    mask2  = mask_np[14,:,:][:,:,np.newaxis]

    fea1 = calc_radio_fea(image1, mask1)
    fea2 = calc_radio_fea(image2, mask1)
    loss_list, var1_list, var2_list = compare_radio_feas(fea1, fea2)
    print("---------------------------------------")
    print(loss_list)
    print("---------------------------------------")
    print(var1_list)
    print("---------------------------------------")
    print(var2_list)
    
if __name__ == "__main__":
    usage()
    