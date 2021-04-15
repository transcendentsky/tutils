import torch
import torchvision
import torch.nn as nn
import numpy as np

def match_inner_product(feature, template):
    feature = feature.permute(0, 2, 3, 1)
    template = template.unsqueeze(1).unsqueeze(1)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-6)
    return cos_similarity

class CompLoss(nn.Module):
    """
    Comparative loss && InfoNCE Loss
    """
    def __init__(self, mode="infonce", c_size=3, p=0.1):
        super(CompLoss, self).__init__()
        self.name = ""
        self.func = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.softmax = torch.nn.Softmax(dim=0)
        self.mode = mode
        self.c_size = c_size
        self.p = p

    def forward(self, x, loc, heatmap, *args, **kwargs):
        if self.mode == "infonce":
            return self._infonce_loss_from_map(x, loc, heatmap, *args, **kwargs)
        elif self.mode == "cos_sim":
            return self._infonce_loss_cossim_from_map(x, loc, heatmap, *args, **kwargs)
        elif self.mode == "contrastive":
            return self.heatmap_loss(heatmap)
        else:
            raise NotImplementedError

    def _infonce_cos_sim(self, x, pos, neg):
        """
        InfoNCE via CosineSimilarity
        """
        x = torch.norm(x, dim=0)
        pos = torch.norm(pos, dim=0)
        neg = torch.norm(neg, dim=0)
        upper = torch.sum(self.cos_sim(x, pos))
        lower = torch.sum(self.cos_sim(x, neg))
        result = -torch.log(upper / (lower + upper))
        return result

    def _infonce_loss(self, x, pos, neg):
        """
        InfoNCE loss
            x: template - shape([16])
            pos: positive points - shape([16])
            neg: negative points - shape([16, 784])
        """
        # print("_infonce_loss: shapes ", x.shape, pos.shape, neg.shape)
        x = self.softmax(x)   # normalization via softmax
        pos = self.softmax(pos)
        neg = self.softmax(neg)
        upper = torch.sum(torch.exp(x * pos))
        lower = torch.sum(torch.exp(x.unsqueeze(-1) * neg))
        result = -torch.log(upper / (lower + upper))
        return result 

    def _sample_points_from_map(self, _loc, _map, _c_size=3, p=0.2):
        """
            _loc: (x, y)
            _map: shape=(m,n,channel)
            _c_size: center size, points within center_size will not be selected
            p: proportion of negative points
        """
        assert len(_map.size()) == 3
        _map = _map.permute(1,2,0)
        m, n, c = _map.size()
        mask = torch.rand((m, n))
        mask[  int(np.clip(_loc[0]-_c_size, 0, m)) : int(np.clip(_loc[0]+_c_size, 0, m)), \
            int(np.clip(_loc[1]-_c_size, 0, n)) : int(np.clip(_loc[1]+_c_size, 0, n))  ] = 0
        neg = _map[torch.where(mask > (1-p))]
        # pos = _map[_loc]
        pos = _map[_loc[0], _loc[1], :]
        # print(f"pos ??? : ", pos.shape, _map.shape, _loc)
        neg = neg.permute(1,0)
        return pos, neg


    def _infonce_loss_cossim_from_map(self, x, loc, _map, _c_size=0, p=0):
        _c_size = self.c_size if _c_size <= 0 else _c_size
        p = self.p if p <= 0 else p
        b = _map.size(0)
        loss_list = []
        for i in range(b):
            pos, neg = self._sample_points_from_map(loc[i], _map[i], _c_size, p)
            loss = self._infonce_cos_sim(x[i], pos, neg)
            loss_list.append(loss)
        return torch.mean(torch.stack(loss_list)) 

    def _infonce_loss_from_map(self, x, loc, _map, _c_size=0, p=0):
        """
        Select the point (_loc) as positve and select random points as negative.
        Note: single batch
            _loc: the GT location: (x, y)
            _map: already calculated similarity map
            _c_size: center size, points within center_size will not be selected
            p: proportion of negative points
        """
        _c_size = self.c_size if _c_size <= 0 else _c_size
        p = self.p if p <= 0 else p
        b = _map.size(0)
        loss_list = []
        for i in range(b):
            pos, neg = self._sample_points_from_map(loc[i], _map[i], _c_size, p)
            loss = self._infonce_loss(x[i], pos, neg)
            loss_list.append(loss)
        return torch.mean(torch.stack(loss_list)) 

    def triple_loss(self, x, pos, neg, func=None, eps=0.001):
        """
        A template and a positive and a negative:
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

    def heatmap_loss(self, heatmap, pos_num=5, neg_num=5):
        """
        x:   (b, 1, num_classes)
        pos: (b, 1, num_classes)
        neg: (b, neg_numbers, num_classes)
        """
        x, pos, neg = self._sample(heatmap)
        loss = self._infonce_loss(x, pos, neg)
        return loss

    def _sample(self, heatmap, neg_numbers=9):
        """
        heatmap: (b, num_class, m, n)
        Note: Positive points selection Policy: adjacent points
        Return:
            x:   (b, 1, num_classes)
            pos: (b, 1, num_classes)
            neg: (b, neg_numbers, num_classes)
        """
        raise ValueError("This func is wrong implememnted")
        # b, num, m, n = heatmap.size()
        # loc_x = np.random.randint(0, m)
        # loc_y = np.random.randint(0, n)
        # x = heatmap[:, :, loc_x, loc_y].view(b, 1, num)
        # # Positive points selection Policy: adjacent points
        # pos_loc = np.random.randint(-1,1,2)
        # pos = heatmap[:, :, pos_loc[0], pos_loc[1]].view(b, 1, num)
        # neg_loc_x = np.random.randint(0, m, neg_numbers)
        # neg_loc_y = np.random.randint(0, n, neg_numbers)
        # neg = heatmap[:,:,neg_loc_x, neg_loc_y].view(b,neg_numbers,num)
        # return x, pos, neg

def usage():
    # Usage
    x = torch.rand(2, 16).cuda()
    _map = torch.rand((2, 16, 32, 32)).cuda()
    _loc = [(3,3), (9,9)]
    closs = CompLoss().cuda()
    print(closs.mode)
    loss = closs(x, _loc, _map)
    print(loss)


if __name__ == "__main__":
    x = torch.rand(2, 16).cuda()
    _map = torch.rand((2, 16, 32, 32)).cuda()
    cossim = nn.CosineSimilarity(dim=1, eps=1e-6)
    l1 = match_inner_product(_map, x)
    l2 = cossim(x.unsqueeze(-1).unsqueeze(-1), _map)
    print(l1, l2)