import os
import sys
from PIL import Image
import numpy as np
# import cv2
from tutils import tfilename
from tutils.timg.edge_detection import canny


def test1():
    img = Image.open('./Raw_edge.png').convert('L')
    img_np = np.array(img)
    print("azheazhe: ", img_np[100])

def test2():
    """ Standard Edge """
    dir_pth = "/home/quanquan/oneshot-landmark/dataset/Cephalometric/RawImage/TrainingData/"
    save_pth = "/home/quanquan/oneshot-landmark/dataset/Cephalometric/RawImage/edge/"
    filenames = [x.name for x in os.scandir(dir_pth) if x.name.endswith('.bmp')]
    filenames.sort()
    for filename in filenames:
        file_pth = tfilename(dir_pth, filename)
        img = Image.open(file_pth).convert('RGB')
        img = img.resize((384,384))
        # print(type(img))
        img_np = np.array(img)
        # print(img_np.shape)
        mask = canny(img_np)
        mask = Image.fromarray(mask)
        mask.save(tfilename(save_pth, f"{filename[:-4]}_bg.png"))
        print("Save over: ", tfilename(save_pth, f"{filename[:-4]}_bg.png"))

def test3():
    """ Edge with margins clipped """
    dir_pth = "/home/quanquan/oneshot-landmark/dataset/Cephalometric/RawImage/TrainingData/"
    save_pth = "/home/quanquan/oneshot-landmark/dataset/Cephalometric/RawImage/edge/"
    filenames = [x.name for x in os.scandir(dir_pth) if x.name.endswith('.bmp')]
    filenames.sort()
    for filename in filenames:
        file_pth = tfilename(dir_pth, filename)
        img = Image.open(file_pth).convert('RGB')
        img = img.resize((384,384))
        # print(type(img))
        img_np = np.array(img)
        # print(img_np.shape)
        # import ipdb;ipdb.set_trace()
        # print(img_np.shape)
        mask = canny(img_np)
        assert len(mask.shape) == 2
        margin = int(mask.shape[0] * 0.05)
        m, n = mask.shape
        mask[m-margin:, :] = 0
        mask[:margin, :]   = 0
        mask[:, :margin]   = 0
        mask[:, m-margin:] = 0
        mask = Image.fromarray(mask)
        mask.save(tfilename(save_pth, f"{filename[:-4]}_bg2.png"))
        print("Save over: ", tfilename(save_pth, f"{filename[:-4]}_bg2.png"))

def test4():
    save_pth = "/home/quanquan/oneshot-landmark/dataset/Cephalometric/RawImage/edge/"
    filenames = [x.name for x in os.scandir(save_pth) if x.name.endswith('bg2.png')]
    filenames.sort()
    img = Image.open(tfilename(save_pth, filenames[0])).convert("L")
    img_np = np.array(img)
    loc = np.where(img_np>0)
    # print(loc)
    # print(img_np[loc[0], loc[1]])
    np.save("testloc.npy",loc)
    data = np.load("testloc.npy")
    print(data[0], data[1])
    import ipdb;ipdb.set_trace()

def test5():
    import argparse
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='', help="name of the run")
    parser.add_argument("--config", default="cconf.yaml", help="default configs")
    parser.add_argument("--test", type=int, default=0, help="Test Mode")
    parser.add_argument("--resume", type=int , default=0)
    parser.add_argument("--pretrain", type=str, default="1") #emb-16-289
    parser.add_argument("--pepoch", type=int, default=0)
    parser.add_argument("--edge", type=int, default=1)
    args = parser.parse_args()

    config = {"tag":"hohoho", "test":1, "dasdas":"dasdas"}
    print(args)
    # print(dict(args))
    print(vars(args))
    # for key, value in vars(args).items():
    #     config[]
    config = {**vars(args), **config}
    print(config)

def test6():
    import yaml
    import yamlloader
    """ Test yamlloader """
    
    with open("conf.yaml") as f:
        config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    print(config)
    with open("conf_dump.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    test6()