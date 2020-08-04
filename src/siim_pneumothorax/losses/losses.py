import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

# Losses based on
# https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, out_flat, mask_flat):
        out_flat = torch.sigmoid(out_flat)
        intersection = (out_flat*mask_flat).sum()
        dice = (2.0*intersection + self.smooth)/(out_flat.sum() + mask_flat.sum() + self.smooth)
        return 1. - dice


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*((1-pt)**self.gamma)*BCE_loss

        return torch.mean(F_loss)

# Mixed Loss between BCE Loss and Dice Loss
class MixedLoss(nn.Module):
    def __init__(self, smooth, beta=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.beta = beta

    def forward(self, input, target):
        loss = self.bce(input, target) + self.dice_loss(input, target)
        return loss.mean()

# Mixed Loss between Focal Loss and Dice Loss
class MixedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, smooth=1, beta=1):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.beta = beta

    def forward(self, input, target):
        loss = self.beta*self.focal(input, target) + self.dice_loss(input, target)
        return loss.mean()