import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLCMLoss(nn.Module):
    """
    pay more attention to pixel pairs.
    """
    def __init__(self):
        super(GLCMLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def pair_matrix(self, x):
        """
        [x1 x2]
        """
        x1 = x[:, :-1, :-1]
        x2 = x[:, 1:, 1:]
        return torch.stack([x1,x2], axis=1)

    def forward(self, x, y):
        """
            shape: b, m, n
        """
        assert len(x.size()) == 3
        x_m = self.pair_matrix(x)
        y_m = self.pair_matrix(y)
        loss = self.mse(x, y)
        return loss / 2.


class GLDMLoss(nn.Module):
    """
    8-neighbor GLDM, focus on nearby pixels within range.
    """
    def __init__(self):
        super(GLDMLoss, self).__init__()
        self.threshold = 1.
        self.mse = nn.MSELoss(reduce=False)

    def neighbor_matrix(self, x):
        """
        [ x1 x2 x3 ]
        [ x4 xc x6 ]
        [ x7 x8 x9 ]
        (di = xi - \mu)
        if di < threshold, di = 0 , else di = 1
        """
        xc = x[:, 1:-1,1:-1]
        x1 = x[:, :-2, :-2] - xc
        x2 = x[:, 1:-1,:-2] - xc
        x3 = x[:, 2:,  :-2] - xc
        x4 = x[:, :-2, 1:-1] - xc
        # xc= x[:, 1:-1,1:-1]
        x6 = x[:, 2:,  1:-1] - xc
        x7 = x[:, :-2 ,2: ] - xc
        x8 = x[:, 1:-1,2: ] - xc
        x9 = x[:, 2: , 2: ] - xc
        xm = torch.stack([x1,x2,x3,x4, x6,x7,x8,x9], axis=1)
        mask = torch.where(xm <= self.threshold, 0., 1.)
        return xm, mask

    def forward(self, x, y):
        assert len(x.size()) == 3, f"Got shape {x.shape}"
        x_m, x_mask = self.neighbor_matrix(x)
        y_m, y_mask = self.neighbor_matrix(y)
        mask = torch.where(x_mask==y_mask, 0., 1.)
        loss = self.mse(x_m, y_m)
        loss = torch.sum(mask*loss)
        return loss / 8.

def rgb_to_grey(img:torch.Tensor) -> torch.Tensor:
    """
    convert RGB image to Grey level image: 
    Grey scale = .2126 * R^gamma + .7152 * G^gamma + .0722 * B^gamma
    """
    img = img[:,0,:,:]*.2126 + img[:,0,:,:]*.7152 + img[:,0,:,:]*.0722
    return img

def usage():
    glcmlossfn = GLCMLoss()
    gldmlossfn = GLDMLoss()
    images = torch.randn(2, 64, 64, requires_grad=True)
    gts = torch.rand(2, 64, 64, requires_grad=True)
    loss1 = glcmlossfn(images, gts)
    loss2 = gldmlossfn(images, gts)
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    usage()