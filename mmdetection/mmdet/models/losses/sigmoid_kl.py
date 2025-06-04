from mmdet.registry import MODELS
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .utils import weighted_loss


@weighted_loss
def sigmoid_kl(pred, target, fg_weight, bg_weight) -> Tensor:
    assert len(target) == 2
    teacher_pred, labels = target

    pred_sigmoid = torch.cat([pred.sigmoid().unsqueeze(-1),
                              1-pred.sigmoid().unsqueeze(-1)], -1)
    teacher_pred_sigmoid = torch.cat([teacher_pred.sigmoid().unsqueeze(-1),
                                      1-teacher_pred.sigmoid().unsqueeze(-1)], -1)
    loss_cls_kd = F.kl_div(torch.log(pred_sigmoid.clamp_min(1e-8)),
                           teacher_pred_sigmoid.detach(), reduction='none').reshape(-1, pred.size(1)*2).sum(1)
    labels = labels.reshape(-1)
    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((labels >= 0) & (labels < bg_class_ind)).float()
    loss_cls_kd = loss_cls_kd * pos * fg_weight + loss_cls_kd * (1 - pos) * bg_weight
    loss_cls_kd = loss_cls_kd.mean(0)
    return loss_cls_kd


@MODELS.register_module()
class SigmoidKL(nn.Module):
    def __init__(self,
                 fg_weight: float = 1.0,
                 bg_weight: float = 1.5,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        super(SigmoidKL, self).__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_bbox = self.loss_weight * sigmoid_kl(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor,
            fg_weight=self.fg_weight, bg_weight=self.bg_weight)
        return loss_bbox

