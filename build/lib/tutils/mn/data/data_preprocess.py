# coding: utf-8
"""
    Histogram 直方图调整:?? 下面的这个和我想象中的不一样

    Z-Norm: 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import SimpleITK as sitk


# ------------------ Spacing  ------------------
def resampleImage(Image:sitk.SimpleITK.Image, SpacingScale=None, NewSpacing=None, NewSize=None, Interpolator=sitk.sitkLinear)->sitk.SimpleITK.Image:
    """
    Author: Pengbo Liu
    Function: resample image to the same spacing
    Params:
        Image, SITK Image
        SpacingScale / NewSpacing / NewSize , are mutual exclusive, independent.
    
    def usage():
        image = resampleImage(image, NewSpacing=new_spacing)
    """
    Size = Image.GetSize()
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()

    if not SpacingScale is None and NewSpacing is None and NewSize is None:
        NewSize = [int(Size[0]/SpacingScale),
                   int(Size[1]/SpacingScale),
                   int(Size[2]/SpacingScale)]
        NewSpacing = [Spacing[0]*SpacingScale,
                      Spacing[1]*SpacingScale,
                      Spacing[2]*SpacingScale]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0], Spacing[1], Spacing[2], NewSpacing[0], NewSpacing[1],  NewSpacing[2]))
    elif not NewSpacing is None and SpacingScale is None and NewSize is None:
        NewSize = [int(Size[0] * Spacing[0] / NewSpacing[0]),
                   int(Size[1] * Spacing[1] / NewSpacing[1]),
                   int(Size[2] * Spacing[2] / NewSpacing[2])]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0], Spacing[1], Spacing[2], NewSpacing[0], NewSpacing[1], NewSpacing[2]))
    elif not NewSize is None and SpacingScale is None and NewSpacing is None:
        NewSpacing = [Spacing[0]*Size[0] / NewSize[0],
                      Spacing[1]*Size[1] / NewSize[1],
                      Spacing[2]*Size[2] / NewSize[2]]
        print('Spacing old: [{:.3f}, {:.3f}, {:.3f}] Spacing new: [{:.3f}, {:.3f}, {:.3f}]'.format(Spacing[0],Spacing[1],Spacing[2],NewSpacing[0],NewSpacing[1],NewSpacing[2]))


    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    Resample.SetOutputSpacing(NewSpacing)
    Resample.SetInterpolator(Interpolator)
    NewImage = Resample.Execute(Image)

    return NewImage    


def bad_hist_adjust_example():
    def custom_hist(gray,name):
        h, w = gray.shape
        hist = np.zeros([256], dtype=np.int32)
        for row in range(h):
            for col in range(w):
                pv = gray[row, col]
                hist[pv] += 1

        y_pos = np.arange(0, 256, 1, dtype=np.int32)
        plt.bar(y_pos, hist, align='center', color='r', alpha=0.5)
        plt.xticks(y_pos, y_pos)
        plt.ylabel('Frequency')
        plt.title('Histogram')
        # plt.show()
        plt.savefig(name)

    path = "/home/quanquan/code/tutils/tutils/paper_writing/corgi1.jpg"
    src = cv2.imread(path)
    assert os.path.exists(path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    cv2.imwrite("gray.jpg",gray)
    # cv2.imshow("input", gray)
    dst = cv2.equalizeHist(gray)
    cv2.imwrite("dst.jpg",dst)
    # cv2.imshow("eh", dst)

    custom_hist(gray, "gray_h.jpg")
    custom_hist(dst, "dst_h.jpg")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()