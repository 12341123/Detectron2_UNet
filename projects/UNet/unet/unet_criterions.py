import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossEntropy_DiceLoss', 'DiceLoss']

# Used by Unet++
class CrossEntropy_DiceLoss(nn.Module):

    def __init__(self, weight_CE=0.5, weight_DL=1, smooth=1):

        """
        Args: 
        weight_CE, weight_DC: Unet++ combines both crossentropy loss and dice loss, this two
                   factors is used to balance two kind of weights. Loss is calculated as:
                                Loss = weight_CE * CE + weight_DC * DC
                   In the paper, weight_CE = 1/2, weight_DL = 1
        smooth: used when calculate DC = (2*|X inter Y| + smooth) / (|X| + |Y| + smooth) to
                avoid division by zero, default is 1.
        """
        super().__init__()

        self.weight_CE = weight_CE
        self.weight_DL = weight_DL
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    
    def forward(self, x, targets):

        CE = self.cross_entropy(x, targets)
        DL = self.dice_loss(x, targets)
        return self.weight_CE * CE + self.weight_DL * DL


# Implementation of multi-class dice-loss
class DiceLoss(nn.Module):

    def __init__(self, smooth=1):

        super().__init__()
        self.smooth = smooth

    
    def forward(self, pred, target):
        
        """
        Input pred: [N, C, H, W] without softmax
        Input target: [N, H, W] 8-bit class map
        First convert target to [N, C, H, W] onehot map, then compute binary DiceLoss for
        each class, and take average.
        """
        pred_softmax = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_onehot = self.to_onehot(num_classes, target)

        total_loss = 0
        # Calculate DiceLoss per-class
        for class_ind in range(num_classes):
            total_loss += self.binary_diceloss(
                pred=pred_softmax[:, class_ind],
                target=target_onehot[:, class_ind]
            ) 
        return total_loss / (num_classes - 1)
        

    def binary_diceloss(self, pred, target):
        
        # Input: pred [N, H, W] after softmax, target [N, H, W]

        batch_size = pred.size(0)
        X = pred.view(batch_size, -1)
        Y = target.view(batch_size, -1)
        intersection = (X * Y).sum()

        return 1 - (2. * intersection + 1) / (X.sum() + Y.sum() + 1)


    def to_onehot(self, num_classes, target):
        
        # Convert [N, H, W] to [N, C, H, W] in onehot form.
        try:
            batch, height, width = target.shape
        except:
            raise ValueError('Invalid input target shape!')
        
        output = []
        for class_ind in range(num_classes):
            out = torch.zeros(target.shape)
            out = (target == class_ind) + 0
            output.append(out)
        output = torch.cat(output, dim=1).view(batch, num_classes, height, width)

        return output


    


