import numpy as np 
import torch
import time
from pathlib import Path

from multiprocessing import Process, Queue
from PIL import Image
from torchvision.transforms import ToPILImage

from PIL import Image, ImageDraw, ImageFont

to_PIL = ToPILImage()

def pred2gt(pred):
    if len(pred) != 2: 
        return pred
    # Convert predicts to GT format
    # pred :  list[ c(y) ; c(x) ]
    out = list()
    for i in range(pred[0].shape[-1]):
        out.append([int(pred[1][i]), int(pred[0][i])])
    return out

def visualize(img:torch.Tensor, landmarks, red_marks, ratio=0.01, draw_line=True):
    """
    # img : tensor [1, 3, h, w]
    landmarks: [ [x,y], [x,y], ... ]  nd.ndarray
    red_marks: [ [x,y], [x,y], ... ]  nd.ndarray
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    h, w = img.shape[-2], img.shape[-1]
    Radius_Base = int(min(h, w) * ratio)
    img = (img - img.min()) / (img.max() - img.min())
    img = img.cpu()
    num_landmarks = len(pred2gt(landmarks))
    # Draw Landmark
    # Green [0, 1, 0] Red [1, 0, 0]
    Channel_R = {'Red': 1, 'Green': 0, 'Blue': 0}
    Channel_G = {'Red': 0, 'Green': 1, 'Blue': 0}
    Channel_B = {'Red': 0, 'Green': 0, 'Blue': 1}
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)

    landmarks = pred2gt(landmarks)

    image = to_PIL(img[0])
    draw = ImageDraw.Draw(image)
    for i, landmark in enumerate(landmarks):
        red_id = red_marks[i]
        Radius = Radius_Base
        draw.rectangle((red_id[0]-Radius, red_id[1]-Radius,\
            red_id[0]+Radius, red_id[1]+Radius), fill=red)
        draw.rectangle((landmark[0]-Radius, landmark[1]-Radius, \
            landmark[0]+Radius, landmark[1]+Radius), fill=green)
        if draw_line:
            draw.line([tuple(landmark), (red_id[0], red_id[1])], fill='green', width=0)
    
    return image

def save_img(image:Image.Image, path:str):
    image.save(path)
