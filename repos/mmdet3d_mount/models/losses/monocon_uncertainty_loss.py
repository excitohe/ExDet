import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class MonoConUncertaintyLoss(nn.Module):

    def __init__(self, mode='laplacian', loss_weight=1.0):
        super(MonoConUncertaintyLoss, self).__init__()
        assert mode in ['laplacian', 'gaussian']
        self.mode = mode
        self.loss_weight = loss_weight

    def forward(self, source, target, logvar):
        logvar = logvar.flatten()
        source = source.flatten()
        target = target.flatten()

        diff = torch.abs(source - target)
        if self.mode == 'laplacian':
            loss = 1.4142 * torch.exp(-logvar) * diff + logvar
        elif self.mode == 'gaussian':
            loss = 0.5 * torch.exp(-logvar) * diff**2 + 0.5 * logvar
        else:
            raise ValueError(f'Unsupport uncertainty mode {self.mode}')
        return loss.mean() * self.loss_weight