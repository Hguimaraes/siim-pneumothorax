import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Dice_metric(nn.Module):
    def __init__(self, smooth=1, threshold=0.5):
        super(Dice_metric, self).__init__()
        self.threshold = threshold
        self.eps = 1e-4
        self.smooth = smooth

    def forward(self, out_flat, mask_flat):
        with torch.no_grad():
            out_flat = torch.sigmoid(out_flat)
            intersection = (out_flat*mask_flat).sum()
            dice = (2.0*intersection + self.smooth)/(out_flat.sum() + mask_flat.sum() + self.smooth)
            return dice


# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
class IoU_metric(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU_metric, self).__init__()
        self.threshold = threshold
        self.eps = 1e-4

    def forward(self, preds, labels, ignore=None):
        with torch.no_grad():   
            ious = []
            for pred, label in zip(preds, labels):
                intersection = ((label == 1) & (pred == 1)).sum()
                union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
                if not union:
                    iou = 1
                else:
                    iou = float(intersection) / float(union)
                ious.append(iou)
            iou = np.nanmean(ious)    # mean accross images if per_image
            return iou