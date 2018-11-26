import torch
from torch import nn
from torch.nn import functional as F

class Loss:
    def __init__(self, nll_weight=1, dice_weight=1):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.nll_weight = nll_weight

    def __call__(self, outputs, targets):
        loss = self.nll_weight * self.nll_loss(outputs, targets)
        
        if self.dice_weight:
            outputs = torch.sigmoid(outputs)
            eps = 1e-15
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= self.dice_weight * torch.log(2 * intersection / union)

        return loss
