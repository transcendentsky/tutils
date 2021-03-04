import torch
import torchvision
import torch.nn as nn
import numpy as np


class CompLoss:
    """
    Comparative loss
    InfoNCE
    """
    def __int__(self, mode="contrastive"):
        self.name = ""
        self.func = cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = torch.nn.Softmax(dim=2)
        self.mode = mode

    def __call__(self, heatmap):
        if self.mode == "contrastive":
            return self.heatmap_loss(heatmap)
        else:
            raise NotImplementedError

    def triple_loss(self, x, pos, neg, func=None, eps=0.001):
        """
        func: similarity function
        eps: minimum gap
        """
        func = self.func if func is None else func
        # x1 = x.repeat(pos.shape[1], axis=1)
        # x2 = x.repeat(neg.shape[1], axis=1)
        x1 = x.repeat(pos.size(1), axis=1) # tensor
        x2 = x.repeat(neg.size(1), axis=1) # tensor
        s1 = func(x1, pos)
        s2 = func(x2, neg)
        return s2 + eps - s1

    def _contrastive_loss(self, x, pos, neg):
        """
        InfoNCE loss
        """
        x = self.softmax(x)
        pos = self.softmax(pos)
        neg = self.softmax(neg)
        upper = torch.sum(torch.sum(torch.exp(x * pos), dim=2), dim=1)
        lower = torch.sum(torch.sum(torch.exp(x * neg), dim=2), dim=1)
        result = -torch.log(upper / (lower + upper))
        return result # shape: (b)
        # raise NotImplementedError

    def heatmap_loss(self, heatmap, pos_num=5, neg_num=5):
        """
        x:   (b, 1, num_classes)
        pos: (b, 1, num_classes)
        neg: (b, neg_numbers, num_classes)
        """
        x, pos, neg = self.sample(heatmap)
        loss = self._contrastive_loss(x, pos, neg)
        return loss

    def sample(self, heatmap, neg_numbers=9):
        """
        heatmap: (b, num_class, m, n)
        Note: Positive points selection Policy: adjacent points
        Return:
            x:   (b, 1, num_classes)
            pos: (b, 1, num_classes)
            neg: (b, neg_numbers, num_classes)
        """
        b, num, m, n = heatmap.size()
        loc_x = np.random.randint(0, m)
        loc_y = np.random.randint(0, n)
        x = heatmap[:, :, loc_x, loc_y].view(b, 1, num)
        # Positive points selection Policy: adjacent points
        pos_loc = np.random.randint(-1,1,2)
        pos = heatmap[:, :, pos_loc[0], pos_loc[1]].view(b, 1, num)
        neg_loc_x = np.random.randint(0, m, neg_numbers)
        neg_loc_y = np.random.randint(0, n, neg_numbers)
        neg = heatmap[:,:,neg_loc_x, neg_loc_y].view(b,neg_numbers,num)
        return x, pos, neg