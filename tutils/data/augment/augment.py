# coding: utf-8
# coding: utf-8
import torch
import numpy as np
import os
import cv2
import random
import torchvision
from torchvision import transforms
# RandAugment from https://github.com/DeepVoltaire/AutoAugment
from .autoaugment import ImageNetPolicy
# RandAugment from https://github.com/xiaopingzeng/pytorch-randaugment
# pip install git+https://github.com/ildoonet/pytorch-randaugment
from RandAugment import RandAugment
from PIL import Image
import torchvision.transforms.functional as TF

"""

"""
def augment_multi_img(imgs=[], augment=None):
    """
    Just for exampling
    """
    raise NotImplementedError
    params = transforms.RandomResizedCrop.get_params(img, scale=(0.1, 0.5), ratio=(0.8, 1.25))
    patch = TF.crop(img, *params)
    # return None

def partial_augment(img, augment,mask=None):
    """
    img: PIL Image
    return img: PIL Image
    """
    params = transforms.RandomResizedCrop.get_params(img, scale=(0.1, 0.5), ratio=(0.8, 1.25))
    # print(params)
    patch = TF.crop(img, *params)
    img_np = np.array(img)
    img2 = augment(img)
    img2_np = np.array(img2)
    # print("imgnp.shape", img_np.shape)
    img_np[params[0]:params[0]+params[2], params[1]:params[1]+params[3]] = \
        img2_np[params[0]:params[0]+params[2], params[1]:params[1]+params[3]] 
    img = Image.fromarray(img_np)
    return img
    # return img, patch

class Augment:
    def __init__(self, img_size=(256, 256)):
        self.randaugment = torchvision.transforms.Compose([RandAugment(10,10), ])
        print("Augment ")

    def check(self, img, *args, **kwargs):
        post_trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normaliza((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(256,256)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            # torchvision.transforms.ColorJitter(hue= ),
        ])

        self.randaugment(img)
        return img


def torch_transform(images:list, size=(256,256)):
    results = []
    # Random crop
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        images[0], output_size=size)
    for im in images:
        im = TF.crop(im, i, j, h, w)
        results.append(im)

    # Random horizontal flipping
    images = results
    if random.random() > 0.5:
        for im in images:
            im = TF.hflip(im)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    return image, mask

def compress_JPEG_from_path(path, quality=10):
    from io import StringIO
    img = Image.open(path)
    buffer = StringIO.StringIO()
    img.save(buffer, "JPEG", quality=10)
    return buffer.contents()


def gaussian_blur(img, kernel=(3, 3), std=0):
    # Gaussian Blur
    return cv2.GaussianBlur(img, kernel, std)

def add_gaussian_noise(img, mean=0, var=0.001):
    # Gaussian
    # img = np.array(img)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise * 255
    out = np.clip(out, 0, 255)
    return out


def add_light_spot(img, scale=0.2, strength=0.2):
    # Draw n circles on image
    num = random.randint(1, 6)
    for _ in range(num):
        bg = np.zeros_like(img)
        center = (int(random.random() * img.shape[0]), int(random.random() * img.shape[1]))
        radius = (min(img.shape) * scale * 0.7 * random.random())
        color = (255, 255, 255)
        thickness = -1  # -1 means to solid circles, >=0 means circle rings
        extra = cv2.circle(bg, center, radius, color, thickness)
        # img = img + extra * strength
        img = np.where(extra <= 0, img, img * (1 + strength))
        img = np.clip(img, 0, 255)
    return img


def re_resize(img, scale=0.5):
    img = cv2.resize(img, fx=0.5, fy=0.5)
    return cv2.resize(img, fx=2, fy=2)


# ------------------------------------------------------
#             copy from Bring-old-to-life
# ------------------------------------------------------
def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256
    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)
    if (h == ph) and (w == pw):
        return img
    return img.resize((w, h), method)

def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = torchvision.transforms.Scale(256, Image.BILINEAR)(img)
    return torchvision.transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255
    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")
    return hole_img
# --------------------------------------------------------

# -------------  Test Functions  -----------------------
def test_compress():
    contents = compress_JPEG_from_path("/home1/quanquan/code/py_tutorials/medical/corgi1.jpg")
    print(type(contents))
    import ipdb; ipdb.set_trace()
    a = np.array(contents)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    test_compress()