import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    """
        Used for RGB image
            if applied to GreyLevel Image, please set kernel repeat off
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        """
        x: image
        y: gt
        """
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

def display():
    from PIL import Image
    from tutils.timg.print_img import torchvision_save
    from torchvision import transforms
    import torchvision
    img_pth = "./Raw.png"
    img = Image.open(img_pth).convert('RGB')
    img = transforms.ToTensor()(img).unsqueeze(0).cuda()
    img = transforms.Normalize([0], [1])(img)
    print(img.shape)
    edgelossfn = EdgeLoss()
    laplacian = edgelossfn.laplacian_kernel(img)
    print("Save img!!")
    # torchvision_save(laplacian, "./edgeloss_edge.jpg")
    torchvision.utils.save_image(laplacian, "./edgeloss_edge.jpg")


if __name__ == "__main__":
    display()