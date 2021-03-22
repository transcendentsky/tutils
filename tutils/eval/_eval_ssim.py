# # coding: utf-8
# import torch
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# # Y: (N,3,H,W)  

# def cal_tensor_ssim(X, Y):
#     # calculate ssim & ms-ssim for each image
#     ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
#     return ssim_val

# def cal_tensor_msssim(X, Y):
#     ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)
#     return ms_ssim_val

# # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
# # ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
# # ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# def tensor_ssim_module():
#     # reuse the gaussian kernel with SSIM & MS_SSIM. 
#     ssim_module = SSIM(data_range=255, size_average=True, channel=3)
#     ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

#     # ssim_loss = 1 - ssim_module(X, Y)
#     # ms_ssim_loss = 1 - ms_ssim_module(X, Y)

# if __name__ == "__main__":
#     b = torch.ones((3,3,256,256))
#     bb = torch.mean(b)
#     # print("dasdas")
#     print(bb)
#     a = torch.rand_like(b)
#     s1 = cal_tensor_ssim(a, b)
#     print(s1)