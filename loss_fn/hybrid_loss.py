import torch.nn as nn
from loss_fn.adasp_loss import AdaSPLoss
from loss_fn.contrastive_loss import SupConLoss


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, temperature=0.07, adasp_temp=0.04, loss_type='adasp'):
        super(HybridLoss, self).__init__()
        self.contrastive_loss = SupConLoss(temperature)
        self.adasp_loss = AdaSPLoss(temp=adasp_temp, loss_type=loss_type)
        self.alpha = alpha  # Weight for the contrastive loss
        self.beta = beta  # Weight for the AdaSPLoss

    def cross_entropy_one_hot(self, input, target):
        _, labels = target.max(dim=1)
        return nn.CrossEntropyLoss()(input, labels)

    def forward(self, y_proj, y_pred, label, label_vec):
        contrastive_loss = self.contrastive_loss(y_proj.unsqueeze(1), label.squeeze(1))
        adasp_loss = self.adasp_loss(y_proj, label)  # Using y_proj for AdaSPLoss
        entropy_loss = self.cross_entropy_one_hot(y_pred, label_vec)

        # Calculate the total loss
        total_loss = (self.alpha * contrastive_loss +
                      self.beta * adasp_loss +
                      (1 - self.alpha - self.beta) * entropy_loss)

        return total_loss

