"""
    cv2 edge detection descriptors, including Sobel, Canny, Laplace, Scharr, etc.
"""

import cv2
import numpy as np



def canny(img:np.ndarray,
          threshold1:int=50, 
          threshold2:int=150) -> np.ndarray:
    img1 = cv2.GaussianBlur(img,(3,3),0)
    canny = cv2.Canny(img1, 50, 150)
    return canny
    
def usage():
    # Demo / example
    from PIL import Image
    img = Image.open("/home1/quanquan/datasets/film-gen/source/hypophysis-mri-film2/T1-Cyr17H.jpg").convert('RGB')
    print(type(img))
    img_np = np.array(img)
    print(img_np.shape)
    mask = canny(img_np)
    mask = Image.fromarray(mask)
    mask.save("T1-Cyr17H-mask.jpg")
    
if __name__ == "__main__":
    usage()