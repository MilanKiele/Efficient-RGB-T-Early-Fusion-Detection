import torch
from torch import Tensor
from typing import List, Tuple
from mmdet.registry import MODELS
from .retina_head import RetinaHead
from mmdet.utils import (InstanceList, OptInstanceList, ConfigType)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.structures import SampleList
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class KDRetinaHead(RetinaHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                cls_kd_loss: ConfigType = None,
                box_kd_loss: ConfigType = None,
                 **kwargs):
        super(KDRetinaHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs
        )
        self.cls_kd_loss = MODELS.build(cls_kd_loss) if cls_kd_loss is not None else None
        self.kd_reg_decoded_bbox = 'IoULoss' in box_kd_loss.get('type')
        self.box_kd_loss = MODELS.build(box_kd_loss) if box_kd_loss is not None else None

    def loss(self,
             x: Tuple[Tensor],
             out_teacher: Tuple[Tensor],
             teacher_head: nn.Module,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            out_teacher (tuple[Tensor]): The output of teacher.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)
        pseudo_student = teacher_head(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs += (*pseudo_student, *out_teacher)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            pseudoS_cls_scores: List[Tensor],    # pseudo student
            pseudoS_bbox_preds: List[Tensor],
            teacher_cls_scores: List[Tensor],   # teacher
            teacher_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        # KNOWLEDGE DISTILLATION
        kd_losses_cls, kd_losses_bbox = multi_apply(
            self.loss_by_kd_single,
            cls_scores, # student
            bbox_preds,
            pseudoS_cls_scores, #   pseudo student
            pseudoS_bbox_preds,
            teacher_cls_scores, # teacher
            teacher_bbox_preds,
            all_anchor_list,
        )
        losses.update(
            dict(kd_loss_cls=kd_losses_cls, kd_loss_bbox=kd_losses_bbox))

        return losses

    def loss_by_kd_single(self,
                          cls_score: Tensor, bbox_pred: Tensor, # student
                          pseudoS_cls_score: Tensor, pseudoS_bbox_pred: Tensor, # pseudo student
                          teacher_cls_score: Tensor, teacher_bbox_pred: Tensor, # teacher
                          anchors: Tensor) -> tuple:
        # KD: Classification
        ## teacher -> student
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        teacher_cls_score = teacher_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        kd_cls = self.cls_kd_loss(cls_score,
                                  teacher_cls_score)
        ## teacher -> pseudo student
        pseudoS_cls_score = pseudoS_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        kd_cls += self.cls_kd_loss(pseudoS_cls_score,
                                   teacher_cls_score)

        # KD: Regression
        ## teacher -> student
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
        teacher_bbox_pred = teacher_bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
        ## teacher -> pseudo student
        pseudoS_bbox_pred = pseudoS_bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
        if self.kd_reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
            teacher_bbox_pred = self.bbox_coder.decode(anchors, teacher_bbox_pred)
            teacher_bbox_pred = get_box_tensor(teacher_bbox_pred)
            pseudoS_bbox_pred = self.bbox_coder.decode(anchors, pseudoS_bbox_pred)
            pseudoS_bbox_pred = get_box_tensor(pseudoS_bbox_pred)

        kd_bbox = self.box_kd_loss(bbox_pred, teacher_bbox_pred)
        kd_bbox += self.box_kd_loss(pseudoS_bbox_pred, teacher_bbox_pred)
        return kd_cls, kd_bbox






