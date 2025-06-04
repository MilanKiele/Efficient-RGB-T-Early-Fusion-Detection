# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .accuracy import accuracy
from .utils import weight_reduce_loss
from .cross_entropy_loss import CrossEntropyLoss



@MODELS.register_module()
class FocusPickingLoss(CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(FocusPickingLoss, self).__init__(**kwargs)

    def neg_entropy(self, logits):
        probs = F.softmax(logits, -1)
        return probs * F.log_softmax(logits, -1)

    def forward(self,
                cls_score,
                label,
                teacher_logits,
                teacher_is_gts,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        # ce_loss = super(FocusPickingLoss, self).forward(
        #         cls_score,
        #         label,
        #         weight=weight,
        #         avg_factor=avg_factor,
        #         reduction_override=reduction_override,
        #         ignore_index=ignore_index,
        #         **kwargs)
        # cls_loss
        zerolabel = cls_score.new_zeros(cls_score.shape)
        ce_loss = F.binary_cross_entropy_with_logits(
            cls_score, zerolabel, reduction='none').float()

        gtbox_label = label[teacher_is_gts.bool()]
        ce_loss[teacher_is_gts.bool()] = F.binary_cross_entropy_with_logits(
            cls_score[teacher_is_gts.bool()], F.one_hot(gtbox_label, cls_score.size(-1)).float(),
            reduction='none').float()


        # 对于非 GT box 的预测结果进行如下操作，
        # 这是因为 GT box 是在 sampling 阶段添加的，
        # 而 EME 输出的候选框是在Sampling 之前完成的
        # 所以 EME 对 GT box 没有置信度得分。
        non_gtbox_inds = teacher_is_gts != 1
        non_gtbox_label = label[non_gtbox_inds]
        non_gtbox_cls_logits = cls_score[non_gtbox_inds]
        assert teacher_logits.size(0) == non_gtbox_label.size(0)

        zerolabel = non_gtbox_cls_logits.new_zeros(non_gtbox_cls_logits.shape)
        aux_loss = F.binary_cross_entropy_with_logits(
            non_gtbox_cls_logits, zerolabel, reduction='none')

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        if self.use_sigmoid:
            bg_class_ind = cls_score.size(1)
        else:
            bg_class_ind = cls_score.size(1) - 1

        pos = ((non_gtbox_label >= 0) & (non_gtbox_label < bg_class_ind)).nonzero().squeeze(1)  # 正样本的索引
        pos_label = non_gtbox_label[pos].long()   # 正样本的类别

        # Cls loss
        difference = F.softmax(non_gtbox_cls_logits[pos, :-1], -1) - F.softmax(teacher_logits[pos], -1)    # 与FRSKD比较时，使用了detach()
        ce_loss[pos, :-1] = F.binary_cross_entropy_with_logits(
            non_gtbox_cls_logits[pos, :-1] + difference, F.one_hot(non_gtbox_label[pos], bg_class_ind).float(),
            reduction='none').float()
        ce_loss = ce_loss.sum(-1).mean(0)
        # Multi-warm label loss
        pos_onehot_labels = F.one_hot(pos_label, bg_class_ind).float()
        pos_teacher_logits = teacher_logits[pos]    # EME 在正样本上的 logits
        confused_label = pos_teacher_logits > 0
        mwlabel = torch.clamp(confused_label.double() + pos_onehot_labels, 0, 1)
        mwlabel = mwlabel/torch.sum(mwlabel, -1, True)
        mwlabel = torch.cat([mwlabel, mwlabel.new_zeros(mwlabel.size(0), 1)], 1)

        aux_loss[pos] = F.binary_cross_entropy_with_logits(
            non_gtbox_cls_logits[pos], mwlabel.detach(),
            reduction='none').float()

        # Negative entropy loss
        alpha = 0.9
        aux_loss[pos, :-1] = alpha * aux_loss[pos, :-1] + \
                             (1.0 - alpha) * self.neg_entropy(non_gtbox_cls_logits[pos, :-1])  # Focusnet 在正样本上的 logits
        beta = 0.9
        return ce_loss * beta + (1.0 - beta) * aux_loss.sum(-1).mean(0)