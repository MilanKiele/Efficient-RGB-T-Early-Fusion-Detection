# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.registry import MODELS, TASK_UTILS
from .kd_gfl_clip import KDGFLCLIP
from mmdet.structures import OptSampleList, SampleList
from torch import Tensor
import torch.nn as nn
from mmdet.structures.bbox import bbox2roi, get_box_tensor, scale_boxes
from mmdet.models.layers import multiclass_nms
import copy
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType, InstanceList
import torch.nn.functional as F
from mmengine.config import ConfigDict
from ..utils import multi_apply, unpack_gt_instances, empty_instances
from typing import List, Tuple, Optional
import torch
from mmdet.models.task_modules.samplers import SamplingResult
from ..losses import FocusPickingLoss


@MODELS.register_module()
class KDGFLCLIPDetFocusNetCls(KDGFLCLIP):
    def __init__(self,
                 focusnet_config,
                 assigner_config,
                 sampler_config,
                 bbox_roi_extractor,
                 rcnn_train_cfg,
                 **kwargs):
        super(KDGFLCLIPDetFocusNetCls, self).__init__(**kwargs)
        self.freeze(self)
        self.focusnet = nn.Module()
        self.focusnet.backbone = MODELS.build(focusnet_config['backbone'])
        self.focusnet.backbone.init_weights()
        self.focusnet.neck = MODELS.build(focusnet_config['neck'])
        self.focusnet.head = MODELS.build(focusnet_config['head'])
        self.focusnet.bbox_assigner = TASK_UTILS.build(assigner_config)
        self.focusnet.bbox_sampler = TASK_UTILS.build(
            sampler_config, default_args=dict(context=self))
        self.focusnet.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.focusnet.rcnn_train_cfg = rcnn_train_cfg

    def freeze(self, model: nn.Module):
        for name, module in self.named_children():
            print(f'Freezing {name:<10}')
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super(KDGFLCLIPDetFocusNetCls, self).train(False)
        self.focusnet.train(mode)
        return self

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        self.focusnet.cuda(device=device)
        return super().cuda(device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        self.focusnet.to(device=device)
        return super().to(device)


    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.focusnet.bbox_roi_extractor([x], rois)

        if self.focusnet.training:
            # 随机生成每张图片是否翻转的标志 (0 或 1)，形状为 [B, 1, 1, 1]
            flip_flags = torch.randint(0, 2, (bbox_feats.size(0), 1, 1, 1), device=bbox_feats.device)
            # 利用翻转标志和切片操作实现随机水平翻转
            bbox_feats = bbox_feats * (1 - flip_flags) + bbox_feats.flip(-1) * flip_flags

        cls_features_bk = self.focusnet.backbone(bbox_feats)
        cls_features = self.focusnet.neck(cls_features_bk)
        cls_score = self.focusnet.head(cls_features)

        bbox_results = dict(cls_score=cls_score)
        return bbox_results


    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.focusnet.head.num_classes,
                                     dtype=torch.long)
        label_weights = pos_priors.new_zeros(num_samples)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights

    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
                proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
                for all proposals in a batch, each tensor in list
                has shape (num_proposals, 4) when `concat=False`,
                otherwise just a single tensor has shape
                (num_all_proposals, 4), the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals, 4) when `concat=False`, otherwise just a
                single tensor has shape (num_all_proposals, 4).
        """
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
        return labels, label_weights

    def loss_and_target(self,
                        cls_score: Tensor,
                        rois: Tensor,
                        sampling_results,
                        sampling_logits,
                        rcnn_train_cfg,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

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
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)

        if isinstance(self.focusnet.head.loss_cls, FocusPickingLoss):
            label, weight = cls_targets
            sampling_logits = torch.cat(sampling_logits, 0)
            priors_is_gts = []
            for sam_res in sampling_results:
                priors_is_gts.append(
                    torch.cat([
                        sam_res.pos_is_gt,
                        sam_res.pos_is_gt.new_zeros(sam_res.neg_inds.size(0))
                    ]))
            priors_is_gts = torch.cat(priors_is_gts, 0)
            assert len(priors_is_gts) == len(label) == len(weight)
            cls_targets = tuple([label, weight, sampling_logits, priors_is_gts])

        losses = self.focusnet.head.loss(
            cls_score,
            *cls_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_targets)



    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results, sampling_logits) -> dict:
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.loss_and_target(
            cls_score=bbox_results['cls_score'],
            rois=rois,
            sampling_results=sampling_results,
            sampling_logits=sampling_logits,
            rcnn_train_cfg=self.focusnet.rcnn_train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results


    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: SampleList,
             rescale: bool = True):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        with torch.no_grad():
            x = self.extract_feat(batch_inputs)
            outs = self.bbox_head(x)
            eme_results_list = self.bbox_head.predict_by_feat(
                *outs, batch_img_metas=batch_img_metas,
                rescale=False, with_nms=False)

        assert len(eme_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        sampling_logits = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = eme_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.focusnet.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.focusnet.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],)
                # feats=[lvl_feat[i][None] for lvl_feat in [batch_inputs]])
            sampling_results.append(sampling_result)

            eme_logits = rpn_results.logits
            bias = torch.tensor(sampling_result.num_gts)
            pos_inds = sampling_result.pos_inds - bias
            pos_inds = pos_inds[pos_inds >= 0]

            if sampling_result.neg_inds.size(0):
                inds = torch.cat([pos_inds,
                                  sampling_result.neg_inds - bias])
            else:
                inds = pos_inds
            assert inds.max() < len(eme_logits), "the index is not right"
            eme_logits = eme_logits[inds]
            # eme_logits = torch.cat([torch.zeros_like(eme_logits)[:sampling_result.num_gts], eme_logits])
            sampling_logits.append(eme_logits)

        losses = dict()
        # bbox head loss
        bbox_results = self.bbox_loss(batch_inputs, sampling_results, sampling_logits)
        losses.update(bbox_results['loss_bbox'])

        return losses

    def _predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        batch_img_metas: List[dict]) -> InstanceList:
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                img_meta=img_meta,
            )
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            img_meta: dict) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            raise NotImplementedError

        # some loss (Seesaw loss..) may have custom activation
        if self.focusnet.head.use_sigmoid_cls:
            # 当使用 Sigmoid 激活函数时，因为输出维度是 num_classes维度，所以直接做 Sigmoid
            scores = F.sigmoid(cls_score)
        else:
            # 当使用 SoftMax 激活函数时，因为输出维度是 num_classes维度 + 1，所以需要取前 num_classes维度
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
            scores = scores[:, :-1]

        results.scores, results.labels = torch.max(scores, -1)

        return results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList):
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return rpn_results_list, None

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        result_list = self._predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            batch_img_metas=batch_img_metas)
        return result_list, cls_scores


    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        with torch.no_grad():
            x = self.extract_feat(batch_inputs)
            outs = self.bbox_head(x)
            results_list = self.bbox_head.predict_by_feat(
                *outs, batch_img_metas=batch_img_metas,
                rescale=False, with_nms=False)

        results_list_fined, focusnet_logits = self.predict_bbox(
            batch_inputs,
            batch_img_metas,
            results_list)

        result_list_final = []
        for img_id in range(len(batch_img_metas)):

            # _eme_logits = results_list[img_id].logits
            # _focusnet_logits = focusnet_logits[img_id][:, :-1]

            # fused_scores = F.sigmoid(_focusnet_logits + _eme_logits)
            # scores, labels = fused_scores.max(dim=-1)

            results_beforePosPro = InstanceData()
            assert len(results_list[img_id].bboxes) == len(results_list_fined[img_id].labels) == len(results_list_fined[img_id].scores)
            results_beforePosPro.bboxes = results_list[img_id].bboxes

            eme_labels = results_list[img_id].labels
            focusnet_labels = results_list_fined[img_id].labels
            # results_beforePosPro.labels = torch.where(
            #     eme_labels == focusnet_labels,
            #     eme_labels,
            #     eme_labels)

            eme_scores = results_list[img_id].scores
            focusnet_scores = results_list_fined[img_id].scores
            # fused_scores = eme_scores * focusnet_scores
            # fused_scores = (fused_scores - torch.min(fused_scores)) / (torch.max(fused_scores) - torch.min(fused_scores)) * torch.max(eme_scores)
            # results_beforePosPro.scores = torch.where(
            #     eme_labels == focusnet_labels,
            #     fused_scores,
            #     eme_scores
            # )
            # 开始融合策略
            final_bboxes = []
            final_labels = []
            final_scores = []
            final_results_beforePosPro = InstanceData()

            for i in range(len(results_beforePosPro.bboxes)):
                # 获取每个框对应的两个模型的预测信息
                eme_label, eme_score = eme_labels[i], eme_scores[i]
                focusnet_label, focusnet_score = focusnet_labels[i], focusnet_scores[i]

                # 融合策略
                if eme_label == focusnet_label:
                    # 标签一致，取eme_scores作为置信度
                    final_bboxes.append(results_beforePosPro.bboxes[i])
                    final_labels.append(eme_label)
                    final_scores.append(eme_score)
                else:
                    # 标签不一致，检查eme_scores的置信度
                    if eme_score >= 0.3:
                        # 保留框，使用eme_scores作为置信度
                        final_bboxes.append(results_beforePosPro.bboxes[i])
                        final_labels.append(eme_label)
                        # final_scores.append(eme_score * (1 - focusnet_score) ** 2)
                        final_scores.append(eme_score * torch.exp(-torch.abs(focusnet_score - eme_score)))
                        # final_scores.append(eme_score * focusnet_score)
                    else:
                        # 删除框
                        continue

            # 转换结果为张量
            final_bboxes = torch.vstack(final_bboxes) if final_bboxes else results_list[img_id].bboxes
            final_labels = torch.stack(final_labels) if final_labels else results_list[img_id].labels
            final_scores = torch.stack(final_scores) if final_scores else results_list[img_id].scores

            final_results_beforePosPro.bboxes = final_bboxes
            final_results_beforePosPro.scores = final_scores
            final_results_beforePosPro.labels = final_labels


            img_meta = batch_img_metas[img_id]
            results = self.bbox_head._bbox_post_process(
                results=final_results_beforePosPro,
                cfg=self.bbox_head.test_cfg,
                rescale=rescale,
                with_nms=True,
                img_meta=img_meta,
            )
            result_list_final.append(results)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, result_list_final)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None):

        with torch.no_grad():
            eme_results = super()._forward(batch_inputs, batch_data_samples)

        results = ()
        proposals = [rpn_results.bboxes for rpn_results in eme_results]
        rois = bbox2roi(proposals)
        # bbox head
        bbox_results = self._bbox_forward(batch_inputs, rois)
        results = results + (bbox_results['cls_score'],)
        return results