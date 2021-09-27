import numpy as np
import cv2
import torch
from PIL import Image
from tutils.tutils import d
import torchvision


def save_image(img, fname="tmp/tmp.png"):
    """
        img type: Numpy, Tensor, PIL image

        Convert Img to tensor, and use torchvision.utils.save_image()
    """
    if isinstance(img, torch.Tensor):
        torchvision_save(img, fname)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        elif len(img.shape) == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        assert len(img.shape) == 4, f"Write image shape ERROR, Got img.shape={img.shape}, fname={fname}"
        img = img.transpose((0,2,3,1))
        torchvision_save(torch.Tensor(img), fname)
    elif isinstance(img, Image.Image):
        pil_save(img, fname)



def torchvision_save(tensor:torch.Tensor, fname) -> None:
    """
    Recommended: 
    tensor: [b, 3, m,n], multiple images be saved in one file.
    Tips: this function times 255 to tensor
    """
    if torch.max(tensor) > 1.0:
        print(f"[DEBUG] tensor.max() from {tensor.max()} to {1.0}, fname={fname}")
        tensor = tensor / tensor.max()
    if tensor.min() < 0.0:
        print(f"[DEBUG] tensor.min() from {tensor.min()} to {0.0}, fname={fname}", )
        tensor = tensor.clamp(0, 1.0)

    # assert torch.max(tensor) <= 1.0 and torch.min(tensor) >= 0.0, f"Error, got Max:{torch.max(tensor)}, and Min:{torch.min(tensor)}"
    torchvision.utils.save_image(tensor, fname)
    
def pil_save(img:Image.Image, fname) -> None:
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    img.save(fname)


def cv_save(img:np.ndarray, fname:str) -> None:
    img = img.astype(np.uint8)
    cv2.imwrite(fname, img)

