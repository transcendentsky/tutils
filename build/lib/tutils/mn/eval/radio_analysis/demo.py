"""
    This file is almost same like example_hellofeatures.py
"""
from __future__ import print_function

import numpy
import SimpleITK as sitk
import six

from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape


def demo():
    imageName, maskName = getTestCase('brain1')
    image = sitk.ReadImage(imageName)
    mask = sitk.ReadImage(maskName)
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
    ###
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
    results = firstOrderFeatures.execute()
    print(results)
    ###
    shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
    shapeFeatures.enableAllFeatures()
    results = shapeFeatures.execute()
    ###
    glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    ###
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
    glrlmFeatures.enableAllFeatures()
    results = glrlmFeatures.execute()
    ###
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
    glszmFeatures.enableAllFeatures()
    results = glszmFeatures.execute()
