import torch
import torch.nn as nn
import torch.nn.functional as F


# loss function
class contrastive_loss(nn.Module):
    """contrastive_loss."""

    def __init__(self):
        super(contrastive_loss, self).__init__()

    def forward(self, input, output, adversarial):
        x = F.l1_loss(output, adversarial)
        y = F.l1_loss(output, input)
        loss = x / y
        return loss


