import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class DimAwareL1Loss(nn.Module):
    """ MonoCon special function to compute the IoU loss (1-IoU)
        of axis aligned bounding boxes.
    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(DimAwareL1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, source, target, dimension):
        dimension = dimension.clone().detach()

        loss = torch.abs(source - target)
        loss /= dimension

        with torch.no_grad():
            compensation_weight = F.l1_loss(source, target) / loss.mean()
        loss *= compensation_weight

        return loss.mean() * self.loss_weight