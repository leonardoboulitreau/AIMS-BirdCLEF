import torch.nn as nn
import torch
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)

        

        tmp = targets * self.alpha * (1. - probas)**self.gamma * bce_loss
        smp = (1. - targets) * probas**self.gamma * bce_loss
        
        loss = tmp + smp
        loss = loss.mean()
        return loss
