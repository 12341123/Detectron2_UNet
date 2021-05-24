import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import *
from .unet_heads import *
from .activations import get_activation
from detectron2.modeling import Backbone, BACKBONE_REGISTRY

__all__ = ['build_unet_backbone']

class UNet(Backbone):
    
    def __init__(self, channels=[64, 128, 256, 512, 1024], activation='relu'):

        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            get_activation(activation),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            get_activation(activation)
        )
        down_stage = []
        up_stage = []
        in_channel = channels[0]
        for out_channel in channels[1:]:
            down_stage.append(UnetBlockDown(in_channel, out_channel, activation=activation))
            in_channel = out_channel

        in_channel = channels[-1]
        for channel in channels[::-1][1:]:
            out_channel = (channel + in_channel // 2) // 2
            up_stage.append(UnetBlockUp(in_channel, out_channel, channel, activation=activation))
            in_channel = out_channel

        self.out_shape = out_channel
        self.down_stage = nn.Sequential(*down_stage)
        self.up_stage = nn.Sequential(*up_stage)


    def forward(self, x):

        x = self.stem(x)
        saved_features = [x]
        for ind in range(len(self.down_stage) - 1):
            x = self.down_stage[ind](x)
            saved_features.append(x)

        x = self.down_stage[-1](x)
        for ind in range(len(self.up_stage)):
            x = self.up_stage[ind](x, saved_features[-ind-1])

        return x

    # Required for compatibility of Detectron2
    def output_shape(self):
        return self.out_shape


@BACKBONE_REGISTRY.register()
def build_unet_backbone(cfg, input_shape):
    return UNet(
                channels=cfg.MODEL.BACKBONE.UNET_CHANNELS,
                activation=cfg.MODEL.BACKBONE.ACTIVATION,
                )