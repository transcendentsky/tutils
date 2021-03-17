""" 
We apply this function from https://github.com/photosynthesis-team/piq 
- pip install piq

Implementation bellow seems not to be correct, according to this issue: 
    https://github.com/scikit-image/scikit-image/issues/5192 , and 
    https://github.com/Po-Hsun-Su/pytorch-ssim/issues/35
https://github.com/Po-Hsun-Su/pytorch-ssim (X)
"""

from piq import ssim, SSIMLoss
# For More Metrice, please refers to https://github.com/photosynthesis-team/piq

def ssim_loss(data_range=1.):
    assert type(data_range) == float
    return SSIMLoss(data_range)


def usage():
    import torch
    from piq import ssim, SSIMLoss

    x = torch.rand(4, 3, 256, 256, requires_grad=True)
    y = torch.rand(4, 3, 256, 256)

    ssim_index: torch.Tensor = ssim(x, y, data_range=1.)

    # loss = SSIMLoss(data_range=1.)
    loss = ssim_loss(data_range=1.)
    output2 = loss(x,y)
    output: torch.Tensor = loss(x, x)
    print(output.item())
    output.backward()

if __name__ == "__main__":
    usage()
    