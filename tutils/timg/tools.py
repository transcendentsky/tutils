from ..tutils import tfilename, p
from .print_img import print_img_auto
import torch


script_config = {
    ''
}

# Code for paste:
#
# model = netG_B
# img_list = [{'name': 'input', 'img_type': 'img', 'img': img[0]}]
# img_list += [{'name': 'recon', 'img_type': 'img', 'img': recons[0]}]
# tt_save_model_imgs(img_list=img_list, model=netG_B, optimizer=optimizer,
#                    output_dir=output_dir, run=run, name=name, epoch=epoch, idx=i)
# tt_save_model_imgs(img_list, model, optimizer, output_dir, run, name, epoch, idx=i)
1
def tt_save_model_imgs(img_list, model, optimizer ,output_dir, run, name, epoch, idx):
    """
    img_list: [{'img':img, 'img_type':'img_type', 'name':'input/recon'}]
    model: model
    optimizer: optimizer
    output_dir: output_dir
    run: running flags
    name: model name
    epoch: epoch
    idx: batch index or other index
    """
    tt_save_model(model=model, optimizer=optimizer,output_dir=output_dir, run=run, name=name, epoch=epoch, idx=idx)
    tt_print_imgs(img_list=img_list, output_dir=output_dir, run=run, name=name, epoch=epoch, idx=idx)


def tt_save_model(model, optimizer,output_dir, name, run, epoch, idx):
    """
    model: model
    optimizer: optimizer
    output_dir: output_dir
    run: running flags
    name: model name
    epoch: epoch
    idx: batch index or other index
    """
    if model is None or optimizer is None:
        p("[tt_save_model] No model or optimizer ")
        return
    state_dict = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        'name': name,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    fname = tfilename(output_dir, "checkpoints", name, run, str(epoch), f"model_{idx}.pth")
    torch.save(state_dict, fname)
    p(f"[tt_save_model] save model {name} to: {fname}")


def tt_print_imgs(img_list:list, output_dir:str, name:str, run:str, epoch:int, idx:int):
    """
    img_list: example:
        [{'img':img, 'img_type':'img_type', 'name':'input/recon'}]
    """
    if img_list is None or len(img_list) == 0:
        p('[tt_print_imgs] No img to be saved')
        return
    for item in img_list:
        fname = tfilename(output_dir, "imgs", name, run, str(epoch), f"{item['name']}_{idx}.jpg")
        print_img_auto(img=item['img'], img_type=item['img_type'], fname=fname)
        p(f"[tt_print_imgs] save img of {item['name']} to: {fname}")


# if __name__ == '__main__':
#     tt_save_model_imgs(img_list, model, optimizer ,output_dir, run, name, epoch, idx)