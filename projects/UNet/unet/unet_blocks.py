import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import get_activation

__all__ = [ 'UnetBlockDown', 
            'UnetBlockUp', 
            'ResUnetBlockDown',
            'ResUnetBlockUp',
        ]


# Used for Unet, Unet++
class UnetBlockDown(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu'):

        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = get_activation(activation)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    
# Used for Unet, Unet++
class UnetBlockUp(nn.Module):

    def __init__(self, in_channels, out_channels, concat_channel, activation='relu'):

        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                          kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + concat_channel, 
                               out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = get_activation(activation)

    def forward(self, x_in, x_saved):
        
        x_in = self.up_conv(x_in)

        # see https://github.com/milesial/Pytorch-UNet
        diffY = x_saved.size()[2] - x_in.size()[2]
        diffX = x_saved.size()[3] - x_in.size()[3]
        x_in = F.pad(x_in, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_saved, x_in], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x 

# Used for Res-Unet
class ResUnetBlockDown(nn.Module):

    # Notice the difference, UnetBlockDown is pool - conv3x3 - bn - relu - conv3x3 - bn - relu
    # ResUnetBlockDown is conv3x3 - bn - relu - conv3x3 - bn - relu - pool - addition - relu

    def __init__(self, in_channels, out_channels, activation='relu'):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
    
    def forward(self, x):
        
        shortcut = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool(out)

        diffY = out.size()[2] - shortcut.size()[2]
        diffX = out.size()[3] - shortcut.size()[3]
        shortcut = F.pad(shortcut, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        return self.activation(shortcut + out)

# Used for ResUnet
class ResUnetBlockUp(nn.Module):
    
    # ResUnetBlockUp: conv3x3 - bn - relu - conv3x3 - bn - relu - upconv2x2 - relu - addition - relu

    def __init__(self, in_channels, concat_channels, activation='relu'):

        super().__init__()

        combined_inc = in_channels + concat_channels
        self.conv1 = nn.Conv2d(combined_inc, combined_inc // 2, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(combined_inc // 2)
        self.conv2 = nn.Conv2d(combined_inc // 2, combined_inc // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(combined_inc // 2)
        self.activation = get_activation(activation)
        self.up_conv = nn.ConvTranspose2d(combined_inc // 2, combined_inc // 4, 
                                          kernel_size=2, stride=2)

        self.shortcut = nn.ConvTranspose2d(combined_inc, combined_inc // 4,
                                          kernel_size=2, stride=2)

    def forward(self, x_in, x_saved):

        # see https://github.com/milesial/Pytorch-UNet for padding issues
        diffY = x_saved.size()[2] - x_in.size()[2]
        diffX = x_saved.size()[3] - x_in.size()[3]
        x_in = F.pad(x_in, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_saved, x_in], dim=1)

        shortcut = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.up_conv(out)

        return self.activation(shortcut + out)



