import numpy as np
from ..tutils import  generate_name, tfilename
import cv2
import torch
from PIL import Image
from tutils.tutils import d
import torchvision

# def print_img_with_reprocess(img, img_type, fname=None):
#     # print("Printing Imgs with Reprocess")
#     if type(img) is torch.Tensor:
#         img = img.transpose(0, 1).transpose(1, 2)
#         img = img.detach().cpu().numpy()
#     assert np.ndim(img) <= 3
#     img = reprocess_auto(img, img_type=img_type)
#     print_img_np(img, img_type, fname=fname)
def torchvision_save(tensor:torch.Tensor, fname):
    """
    Recommended: 
    tensor: [b, 3, m,n], multiple images be saved in one file.
    Tips: this function times 255 to tensor
    """
    assert torch.max(tensor) <= 1.0 and torch.min(tensor) >= 0.0, f"Error, got Max:{torch.max(tensor)}, and Min:{torch.min(tensor)}"
    torchvision.utils.save_image(tensor, fname)
    
def pil_save(img:Image.Image, fname):
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    img.save(fname)

def print_img_auto(img, img_type='ori', is_gt=True, fname=None):
    # print("[Warning] Pause to use print img, "); return
    if type(img) is torch.Tensor:
        print_img_tensor(img, img_type, is_gt, fname)
    elif type(img) is np.ndarray:
        print_img_np(img, img_type, is_gt, fname)
    elif type(img) is Image.Image:
        print_img_np(np.array(img), img_type, is_gt, fname)
    else:
        raise TypeError("Wrong type: Got {}".format(type(img)))


def print_img_tensor(img_tensor, img_type='ori', is_gt=True, fname=None):
    d("[Printing tensor map] ", img_type)
    if len(img_tensor.size()) == 4:
        img_tensor = img_tensor[0, :,:,:]
    elif len(img_tensor.size()) > 4:
        raise ValueError
    img_tensor = img_tensor.transpose(0, 1).transpose(1, 2)
    img_np = img_tensor.detach().cpu().numpy()
    print_img_np(img_np, img_type, is_gt, fname)


def print_img_np(img, img_type='ori', is_gt=True, fname=None):
    """
    Imgs:
        depth, normal, cmap
        uv, bw, background
        ori, ab
    """
    if img_type.lower() in ["ori", "img", "image"]:
        print_ori(img, is_gt, fname=fname)
    elif img_type.lower() in ["exr", "normal"]:
        print_exr(img, is_gt, fname=fname)
    elif img_type.lower() in ['bg', 'back', 'background']:
        print_bg(img, is_gt, fname=fname)
    else:
        raise TypeError("print_img Error!!!")


# ---------------------  Print IMGS  ------------------------
# -----------------------------------------------------------------------------
def print_exr(cmap, is_gt, epoch=0, fname=None):  # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    cmap = cmap * 255
    cmap = cmap.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow", epoch_text,
                                                                 "exr_" + subtitle + "/cmap_" + generate_name() + ".jpg")
    cv2.imwrite(fname, cmap)


# -----------------------------------------------------------------------------
def print_ori(ori, is_gt, epoch=0, fname=None):
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    ori = ori.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow", epoch_text,
                                                                 "ori_" + subtitle + "/ori_" + generate_name() + ".jpg")
    cv2.imwrite(fname, ori)


def print_bg(background, is_gt, epoch=0, fname=None): # [0,1]
    subtitle = "gt" if is_gt else "pred"
    epoch_text = "epoch{}".format(epoch)
    background = background * 255
    background = background.astype(np.uint8)
    fname = tfilename(fname) if fname is not None else tfilename("imgshow",epoch_text,"bg_"+subtitle+"/bg_"+generate_name()+".jpg")
    cv2.imwrite(fname, background)


def print_large_interpolation(uv, mask, fname):
    uv_size = uv.shape[0]
    expand_size = uv_size * 10

    img_rgb = uv  # [:, :, ::-1]
    # img_rgb[:,:,1] = 1-img_rgb[:,:,1]
    img_rgb[:, :, 0] = 1 - img_rgb[:, :, 0]

    s_x = (img_rgb[:, :, 0] * expand_size)  # u
    s_y = (img_rgb[:, :, 1] * expand_size)  # v
    mask = mask[:, :, 0] > 0.6

    img_rgb = np.round(img_rgb)
    s_x = s_x[mask]
    s_y = s_y[mask]
    index = np.argwhere(mask)
    t_y = index[:, 0]  # t_y and t_x is a map
    t_x = index[:, 1]
    # x = np.arange(expand_size)
    # y = np.arange(expand_size)
    # xi, yi = np.meshgrid(x, y)
    mesh = np.zeros((expand_size, expand_size))


def test_new_img(img_path, output_name):
    # cv2.imread()
    uv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # [0,1]
    uv = uv[:, :, 1:]

    print_img_auto(uv, "uv", fname=output_name)


if __name__ == "__main__":
    import os
    from tutils import tfilename
    print("wuhu")
    #
    # dirname = "/home1/quanquan/datasets/generate/mesh_film_small/uv"
    #
    # for x in os.scandir(dirname):
    #     if x.name.endswith("exr"):
    #         test_new_img(x.path, tfilename("test_img_old", x.name[:-4] + ".jpg"))
    #         print(x.name)
