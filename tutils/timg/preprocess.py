# coding: utf-8
"""
    Histogram 直方图调整:?? 下面的这个和我想象中的不一样

    Z-Norm: 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


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