import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)
        # Dice Loss
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        targets = targets.reshape(-1)

        # Focal Loss
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        focal_loss = - (1 - p_t) ** self.gamma * torch.log(p_t + self.smooth)
        focal_loss = focal_loss.mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss