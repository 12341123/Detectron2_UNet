import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_criterions import *
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

__all__ = ['UnetSemSegHead', 'DeepSupervision', 'get_criterion']

# Single Convolutional Head used for UNet
@SEM_SEG_HEADS_REGISTRY.register()
class UnetSemSegHead(nn.Module):

    def __init__(self, cfg, in_channel):

        super().__init__()
        self.conv = nn.Conv2d(in_channel, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, 
                            kernel_size=3, padding=1)
        self.criterion = get_criterion(cfg.MODEL.SEM_SEG_HEAD.SEM_SEG_LOSS_TYPE)
        
        # For Detectron2 compatibility
        self.size_divisibility = 0
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

    def forward(self, x, targets=None):
        x = self.conv(x)
        # if targets[targets > 2] != []:
            # print(targets)
        if self.training:
            loss = self.criterion(x, targets)
            return None, {"loss_sem_seg": loss}
        else:
            return F.softmax(x, dim=1), {} 


# Multi-depth Output Supervisor Used in UnetPP
# Adjustable depth in inference
@SEM_SEG_HEADS_REGISTRY.register()
class DeepSupervision(nn.Module):
    
    def __init__(self, cfg, in_channel):

        super().__init__()
        
        # 1x1 Conv for each output:
        convs = []
        for channel in in_channel:
            convs.append(
                nn.Conv2d(channel, 
                        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, 
                        kernel_size=1,
                        padding=0,
                        )
            )
        self.convs = nn.Sequential(*convs)
        self.criterion = CrossEntropy_DiceLoss()
        
        # For Detectron2 compatibility
        self.size_divisibility = 0
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

    def forward(self, x, targets=None):
        
        # Input x should be a list of output feature maps 

        prob_maps = []
        for ind, feature in enumerate(x):
            prob_maps.append(self.convs[ind](feature))
        pred = sum(prob_maps) / len(prob_maps)

        if self.training:
            loss = self.criterion(pred, targets)
            return None, {"loss_sem_seg": loss}
        else:
            return F.softmax(pred, dim=1), {}


def get_criterion(criterion):

    if criterion == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Unrecognzied loss type {} !'.format(criterion))
    # Wait for more kind of losses to be added in the future!