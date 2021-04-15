# coding: utf-8
# TODO
#进行图片的复制拼接
from PIL import Image
import numpy as np

def concat_images(image_files, fname):
    b = len(image_files)
    img_np = np.array(image_files[0])
    m, n = img_np.shape[0], img_np.shape[1]
    # canvas setting
    UNIT_WIDTH_SIZE = m
    UNIT_HEIGHT_SIZE = n
    COL = 1
    ROW = b
    SAVE_QUALITY = 50
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW)) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    for row in range(ROW):
        for col in range(COL):
            #对图片进行逐行拼接
            #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            #或四元元组（指定复制位置的左上角和右下角坐标）
            target.paste(image_files[COL*row+col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    target.save(fname, quality=SAVE_QUALITY) #成品图保存