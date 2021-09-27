""" 
We apply this function from https://github.com/photosynthesis-team/piq 
- pip install piq

Implementation bellow seems not to be correct, according to this issue: 
    https://github.com/scikit-image/scikit-image/issues/5192 , and 
    https://github.com/Po-Hsun-Su/pytorch-ssim/issues/35
https://github.com/Po-Hsun-Su/pytorch-ssim (X)
"""

import torch
# For More Metrice, please refers to https://github.com/photosynthesis-team/piq

def SSIMLoss(data_range=1.):
    from piq import SSIMLoss as _SSIMLoss
    """
    For torch.Tensor
    """
    assert type(data_range) == float
    return _SSIMLoss(data_range)


def np_ssim(x, y, data_range=1.):
    from piq import ssim
    x = torch.Tensor(x).unsqueeze(0)
    y = torch.Tensor(y).unsqueeze(0)
    ssim_index: torch.Tensor = ssim(x, y, data_range=1.)
    return ssim_index.numpy()[0]

def usage():
    import torch
    from piq import ssim, SSIMLoss

    x = torch.rand(4,  256, 256, requires_grad=True)
    y = torch.rand(4,  256, 256)

    ssim_index: torch.Tensor = ssim(x, y, data_range=1.)
    ssimvalue = ssim_index.detach().numpy()
    assert type(ssim_index.detach().numpy()) == np.ndarray, type(ssim_index.detach().numpy())
    print(f"ssim_index: ", )

    # loss = SSIMLoss(data_range=1.)
    loss = SSIMLoss(data_range=1.)
    output2 = loss(x,y)
    output: torch.Tensor = loss(x, x)
    print(output.item())
    output.backward()

if __name__ == "__main__":
    usage()
    