import torch
import torchvision
import numpy as np
import torch.nn as nn
import cv2
from tutils.data.augment.gaussian import get_guassian_heatmaps_from_ref


class MyEntropyLoss:
    """
    made for heatmap
    """
    def __init__(self, ref_landmarks=None, ref_image=None, with_ref=False):
        self.ref_landmarks = ref_landmarks
        self.num_classes = 19
        self.ref_image = ref_image
        self.with_ref = with_ref
        if ref_image is not None:
            self.ref_shape = ref_image.shape[:2]
        self.heatmaps = get_guassian_heatmaps_from_ref(self.ref_landmarks, self.num_classes, self.ref_shape)

    def entropy_loss(self, logits):
        if self.with_ref:
            return self.entropy_loss_with_ref(logits)
        else:
            # return self.normal_entropy_loss(logits)
            return self.entropy_based_regularization(logits)

    def normal_entropy_loss(self, logits):
        """
        Function:
            find a similar distribution
            calc the loss
        logits: heatmaps
        classes: number of classes
        """
        raise NotImplementedError
        # TODO: replaced with entropy_based_regularization()
        celoss = nn.CrossEntropyLoss()
        # points = logits.flatten()
        maps = logits
        min_loss = None
        for i in range(self.num_classes):
            loss = None
            for j, map in enumerate(maps):
                label = 1 if i == j else 0
                if loss is None:
                    loss = celoss(map, label)
                else:
                    loss += celoss(map, label)
            if min_loss is None or min_loss > loss:
                min_loss = loss

    def entropy_loss_with_ref(self, logits):
        """
        entropy loss with reference image heatmaps
        """
        # TODO:
        torch.argmax(logits)
        raise NotImplementedError
        return

    def entropy_based_regularization(self, logits, reduction=True):
        """
        logits.shape: (b, num_classes, m, n)
        return entropy: (b, m, n)
        """
        b, num_classes, m, n = logits.size()
        logits = logits.permute(b, m, n, num_classes)
        entropy = torch_entropy(logits)
        if reduction:
            entropy = torch.sum(entropy.view(b, -1))
        return entropy

    def get_weights_from_ref(self):
        """
        We should learn wights from the Reference Image
        ref_heatmap: heatmap of reference image

        """
        ref_maps = []
        for i, ref_map in enumerate(ref_maps):
            # find the areas we should focus, and consider the condition in other maps
            ref_map = np.where(np.repeat(self.heatmaps[i][np.newaxis, :, :], axis=0) > 0, self.heatmaps, 0)
            ref_maps.append(ref_map)
        ref_maps = np.stack(ref_maps, axis=0)  # shape: (19, 19, 800, 640)

        return ref_maps


def torch_entropy(p):
    """
    p.shape: (b,m,n, num_classes) => (b,m,n)
    """
    return torch.distributions.Categorical(probs=p).entropy()
