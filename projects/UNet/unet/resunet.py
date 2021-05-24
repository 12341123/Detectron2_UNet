import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import ResUnetBlockDown, ResUnetBlockUp
from .activations import get_activation
from detectron2.modeling import Backbone, BACKBONE_REGISTRY

__all__ = ['build_resunet_backbone']

class ResUNet(Backbone):

    def __init__(self, channels=[64, 128, 256, 512, 1024], activation='relu'):

        super().__init__()

        # Downsampling stage
        down_stage = []
        in_channel = 3
        for channel in channels[:-1]:

            down_stage.append(
                ResUnetBlockDown(
                    in_channels=in_channel,
                    out_channels=channel,
                    activation=activation
                )
            )
            in_channel = channel

        self.down_stage = nn.Sequential(*down_stage)
    
        # The block in the middle
        self.mid_stage = ResUnetMidStageBlock(channels[-2], channels[-1], activation=activation)
        
        # Upsampling stage
        up_stage = []
        in_channel = channels[-1] // 2
        for ind, channel in enumerate(channels[-2::-1][:-1]):

            up_stage.append(
                ResUnetBlockUp(
                    in_channels=in_channel,
                    concat_channels=channel,
                    activation=activation
                )
            )
            in_channel = (in_channel + channel) // 4
        self.up_stage = nn.Sequential(*up_stage)

        # The special last stage of Res-Unet without shortcut connection
        combined_inc = in_channel + channels[0]
        self.last_stage = nn.Sequential(
            nn.Conv2d(combined_inc, combined_inc // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(combined_inc // 2),
            get_activation(activation),
            nn.Conv2d(combined_inc // 2, combined_inc // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(combined_inc // 2),
            get_activation(activation),
            nn.Conv2d(combined_inc // 2, combined_inc // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(combined_inc // 2),
            get_activation(activation)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        self.out_shape = combined_inc // 2

    def forward(self, x):
        
        saved_features = []
        for block in self.down_stage:
            x = block(x)
            saved_features.append(self.upsample(x))

        x = self.mid_stage(x)
        for ind, block in enumerate(self.up_stage):
            x = block(x, saved_features[-ind-1])

        # last stage, conv after concatenation
        x_saved = saved_features[0]
        diffY = x_saved.size()[2] - x.size()[2]
        diffX = x_saved.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_saved, x], dim=1)
        x = self.last_stage(x)

        return x
    
    def output_shape(self):
        return self.out_shape


# Special block in the middle of the Res-UNet
class ResUnetMidStageBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu'):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        self.up_conv = nn.ConvTranspose2d(out_channels, out_channels//2, 
                                        kernel_size=2, stride=2)
        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels//2,
                                        kernel_size=2, stride=2)
    
    def forward(self, x):

        shortcut = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.up_conv(out)

        return self.activation(shortcut + out)


@BACKBONE_REGISTRY.register()
def build_resunet_backbone(cfg, input_shape):
    return ResUNet(
        channels=cfg.MODEL.BACKBONE.UNET_CHANNELS,
        activation=cfg.MODEL.BACKBONE.ACTIVATION,
    )

# For Testing
if __name__ == '__main__':

    model = ResUNet(channels=[16,32,48,64,80])
    print(model)
    inp = torch.randn((2, 3, 64, 64))
    out = model(inp)
    # print(out.shape)
