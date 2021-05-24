import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import *
from .unet_heads import *
from .activations import get_activation
from detectron2.modeling import Backbone, BACKBONE_REGISTRY

__all__ = ['build_unetpp_backbone']

class UNetPP(Backbone):

    def __init__(self, 
                channels=[16, 32, 64, 128], 
                activation='relu',
                use_subnet=0):
        """
        Args: 
        channels - The output channels of the downsampling stage
        activation - pick from ['relu']
        use_subnet - Use 0 if you want to use 'accurate mode' as described in the paper,
                    that means you take average of output of all subnets. Otherwise, the model
                    is in 'fast mode', use 1 if you want to use subnet UNet++ L1, 2 for
                    UNet++ L2, etc. 
                    This is only used in inference, not in training.
                    For more details, see:
                    https://arxiv.org/abs/1807.10165
        """

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            get_activation(activation),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            get_activation(activation)
        )
        
        # layers = [sublayer_0, sublayer_1, ..., sublayer_n] for n = len(channels)-1
        # each sublayer_i is a nn.Sequential of down/upsample plus 2 convs
        layers = []
        # out_channel has exactly the same shape with layers, recording out channels of layers
        out_channels = [channels]

        for ind, channel in enumerate(channels):

            # First sublayer contains downsamples, other sublayers are all upsample layers
            if ind == 0:
                sub_layer = []
                in_channel = channels[0]
                for ind in range(1, len(channels)):
                    sub_layer.append(UnetBlockDown(
                        in_channel, channels[ind], activation=activation
                    ))
                    in_channel = channels[ind]
                layers.append(nn.Sequential(*sub_layer))
            else:
                sub_layer = []
                sub_channels = []
                # No. of channels that we have to concatenate
                concat_channels = [
                        sum([out_channels[j][i] for j in range(ind)])
                        for i, inc in enumerate(channels[ind:])
                    ]

                for inc, concat_channel in zip(out_channels[ind-1][1:], concat_channels):
                    
                    sub_layer.append(UnetBlockUp(
                        in_channels=inc,
                        out_channels=((inc // 2) + concat_channel) // 2,
                        concat_channel=concat_channel,
                        activation=activation
                    ))
                    sub_channels.append(((inc // 2) + concat_channel) // 2)

                out_channels.append(sub_channels)
                layers.append(nn.Sequential(*sub_layer))
                
        self.layers = nn.Sequential(*layers)
        self.use_subnet = use_subnet
        self.out_shape = [out_channels[i][0] for i in range(1, len(channels))]

    def forward(self, x):
        
        x = self.stem(x)
        # Exactly the same shape with self.layers, saving features
        saved_features = []
        
        for ind in range(len(self.layers)):
            
            if ind == 0:
                sub_feature = [x]
                for block in self.layers[0]:
                    x = block(x)
                    sub_feature.append(x)
                saved_features.append(sub_feature)
            else:
                if self.use_subnet != 0 and ind > self.use_subnet and not self.training:
                    break
                sub_feature = []
                for i, block in enumerate(self.layers[ind]):
                    x_saved = torch.cat([
                        saved_features[j][i] for j in range(ind)
                    ], dim=1)
                    x = block(saved_features[ind-1][i+1], x_saved)
                    sub_feature.append(x)
                saved_features.append(sub_feature)

        out_features = [
            saved_features[i][0] for i in range(1, len(saved_features))
        ]
        del saved_features  # Release memory

        return out_features



    # Change output subnet in inference
    def change_subnet_in_inference(self, subnet):
        self.use_subnet = subnet
    
    # For Detectron2 compatibility
    def output_shape(self):
        return self.out_shape



@BACKBONE_REGISTRY.register()
def build_unetpp_backbone(cfg, input_shape):
    return UNetPP(
        channels=cfg.MODEL.BACKBONE.UNET_CHANNELS,
        activation=cfg.MODEL.BACKBONE.ACTIVATION,
        use_subnet=cfg.MODEL.SEM_SEG_HEAD.UNETPP_USE_SUBNET_IN_INFERENCE,
    )
