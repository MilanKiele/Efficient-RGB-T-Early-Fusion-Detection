# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.registry import MODELS
from .cls_head import ClsHead
from mmdet.models.losses import accuracy
from ..losses import FocusPickingLoss

@MODELS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = MODELS.build(loss)
        self.use_sigmoid_cls = loss.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            # 当使用 Sigmoid 激活函数时，输出 num_classes 维度
            # 无论是使用 sigmoid 还是 softmax 激活函数。标签分配 idx = num_class 永远是背景类
            # 当使用 Sigmoid 激活函数时，QFL 是取出前景类 (label>=0 & label < num_class)，计算sigmoid损失；取出背景类 label == num_class，并令其全为 0。
            # 当使用 SoftMax 激活函数时，直接计算预测和label的交叉熵。
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.fc = nn.Linear(self.in_channels, self.cls_out_channels)


    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_cls_channels(self) -> bool:
        """get custom_cls_channels from loss_cls."""
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_activation(self) -> bool:
        """get custom_activation from loss_cls."""
        return getattr(self.loss_cls, 'custom_activation', False)

    # TODO: Create a SeasawBBoxHead to simplified logic in BBoxHead
    @property
    def custom_accuracy(self) -> bool:
        """get custom_accuracy from loss_cls."""
        return getattr(self.loss_cls, 'custom_accuracy', False)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score

    def loss(self,
             cls_score: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             teacher_logits = None,
             teacher_is_gts=None,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                if isinstance(self.loss_cls, FocusPickingLoss):
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        teacher_logits,
                        teacher_is_gts,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    if self.custom_activation:
                        acc_ = self.loss_cls.get_accuracy(teacher_logits, labels[teacher_is_gts != 1])
                        losses.update(acc_)
                    else:
                        losses['teacher_acc'] = accuracy(teacher_logits, labels[teacher_is_gts != 1])
                else:
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        return losses