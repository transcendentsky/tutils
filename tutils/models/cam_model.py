# coding: utf-8
import cv2
import numpy as np
import torch

# 类的作用
# 1.编写梯度获取hook
# 2.网络层上注册hook
# 3.运行网络forward backward
# 4.根据梯度和特征输出热力图

class ShowGradCam:
    def __init__(self,conv_layer):
        assert isinstance(conv_layer,torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self,module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        _min = np.min(cam)
        _max = np.max(cam)
        cam -= _min
        cam /= np.max(cam)
        return {"cam":cam, "min":_min, "max":_max}

    def show_on_img(self,input_img):
        '''
        write heatmap on target img
        :param input_img: cv2:ndarray/img_pth
        :return: save jpg
        '''
        if isinstance(input_img,str):
            input_img = cv2.imread(input_img)
        img_size = (input_img.shape[1],input_img.shape[0])
        fmap = self.feature_res[0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        cam_info = self.gen_cam(fmap, grads_val)
        cam = cam_info['cam']
        cam = cv2.resize(cam, img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)/255.
        cam = heatmap + np.float32(input_img/255.)
        cam = cam / np.max(cam)*255
        cv2.imwrite('grad_feature.jpg',cam)
        print('save gradcam result in grad_feature.jpg')


def usage():
    import os
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool1(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    def img_transform(img_in, transform):
        """
        将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
        :param img_roi: np.array
        :return:
        """
        img = img_in.copy()
        img = Image.fromarray(np.uint8(img))
        img = transform(img)
        img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
        return img

    def img_preprocess(img_in):
        """
        读取图片，转为模型可读的形式
        :param img_in: ndarray, [H, W, C]
        :return: PIL.image
        """
        img = img_in.copy()
        img = cv2.resize(img,(32, 32))
        img = img[:, :, ::-1]   # BGR --> RGB
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
        ])
        img_input = img_transform(img, transform)
        return img_input

    def comp_class_vec(ouput_vec, index=None):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if not index:
            index = np.argmax(ouput_vec.cpu().data.numpy())
        else:
            index = np.array(index)
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
        one_hot.requires_grad = True
        class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

        return class_vec

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join("cat.png")
    assert os.path.exists(path_img), f"path img: {path_img}"
    # path_net = os.path.join("./net.pth") # in this example not use

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 图片读取；网络加载
    img = cv2.imread(path_img)  # H*W*C
    assert img is not None
    img_input = img_preprocess(img)
    net = Net()

    gradCam = ShowGradCam(net.conv2) #............................. def which layer to show

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # save result
    gradCam.show_on_img(img) #.......................... show gradcam on target pic

if __name__ == '__main__':
    usage()