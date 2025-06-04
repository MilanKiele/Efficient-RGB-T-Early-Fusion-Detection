# Copyright (c) OpenMMLab. All rights reserved.
from .faster_rcnn import FasterRCNN
import torch.nn as nn
import clip
import torch
from torch import Tensor
import copy
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from typing import Tuple
from .retinanet_clip import clip_train_loss, unfold_batch
import torch.nn.functional as F

@MODELS.register_module()
class FasterRCNNCLIP(FasterRCNN):
    classes = {6: ['person', 'car', 'bus', 'motorcycle', 'traffic light', 'truck'], 3: ['person', 'bicycle', 'car']}
    extra_classes = ['building', 'plant', 'sky', 'streetlamp', 'streetlight', 'others']
    def __init__(self, **kwargs):
        super(FasterRCNNCLIP, self).__init__(**kwargs)
        resnet_channel = kwargs.get('neck').get('in_channels')[-1]
        self.num_class = kwargs.get('roi_head').get('bbox_head').get('num_classes')
        self.resnet_fcs = nn.ModuleList([nn.Linear(res_c, self.num_class, bias=False) for res_c in [resnet_channel]])
        self.resnet_mask_convs = nn.ModuleList([
                                                nn.Sequential(nn.Conv2d(res_c, self.num_class, 1, bias=False), nn.ReLU(True),
                                                nn.Conv2d(self.num_class, self.num_class, 3, 1, 1, bias=False)) for res_c in [resnet_channel]])
        # self.clip_model, _ = clip.load("RN50", download_root='/home/zx/rgbx-distillation/code/mmdetection-main/myCodeZoo')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", download_root='/home/zx/rgbx-distillation/code/mmdetection-main/myCodeZoo')
        self.device = self.clip_model.visual.conv1.weight.device
        self.clip_preprocess_mean = torch.tensor(self.preprocess.transforms[-1].mean).view(1, 3, 1, 1).to(self.device)
        self.clip_preprocess_std = torch.tensor(self.preprocess.transforms[-1].std).view(1, 3, 1, 1).to(self.device)
        self.clip_fc = nn.Sequential(nn.Linear(self.num_class, self.num_class//2, bias=False), nn.ReLU(True),
                                     nn.Linear(self.num_class//2, self.num_class // 2, bias=False), nn.ReLU(True),
                                     nn.Dropout(),
                                     nn.Linear(self.num_class//2, self.num_class, bias=False)).float()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of the {c}") for c in self.classes[self.num_class] + self.extra_classes]).to(self.device)

    def normalize_batch_inputs(self, batch_inputs):
        rgb_batch, t_batch = torch.chunk(batch_inputs, 2, dim=1)
        rgb_batch = (rgb_batch * self.data_preprocessor.rgb_std + self.data_preprocessor.rgb_mean) / 255.
        t_batch = (t_batch * self.data_preprocessor.thermal_std + self.data_preprocessor.thermal_mean) / 255.
        assert rgb_batch.min() >= 0.0 and rgb_batch.max() <= 1.0
        assert t_batch.min() >= 0.0 and t_batch.max() <= 1.0
        rgb_batch = (rgb_batch - self.clip_preprocess_mean) / self.clip_preprocess_std
        t_batch = (t_batch - self.clip_preprocess_mean.mean()) / self.clip_preprocess_std.mean()
        return torch.cat([rgb_batch, t_batch], 1)

    def extract_feat(self, batch_inputs: Tensor, return_backbone_feats=False) -> [Tuple[Tensor], Tensor]:
        x = self.backbone(batch_inputs)
        if return_backbone_feats:
            backbone_feats = x

        if self.with_neck:
            x = self.neck(x)
        if return_backbone_feats:
            return x, backbone_feats
        else:
            return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:

        x, backbone_feats = self.extract_feat(batch_inputs, return_backbone_feats=True)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)


        # Backbone Auxiliary Branch
        backbone_logits = []
        backbone_masks = []
        feats = backbone_feats[-1]
        for lidx, feat in enumerate(feats if isinstance(feats, list) else [feats]):
            # image level prediction
            logits = self.avgpool(feat).flatten(1)
            logits = self.resnet_fcs[lidx](logits)
            backbone_logits.append(logits)
            # box level prediction
            masks = self.resnet_mask_convs[lidx](feat)
            backbone_masks.append(masks)

        # CLIP
        ## global
        batch_inputs = self.normalize_batch_inputs(batch_inputs)
        batch_global_inputs = F.interpolate(batch_inputs, self.clip_model.visual.input_resolution, mode='bicubic')
        with torch.no_grad():
            batch_global_rgb_inputs, batch_global_t_inputs = torch.chunk(batch_global_inputs, 2, 1)
            global_rgb_feats = self.clip_model.encode_image(batch_global_rgb_inputs)    # [b, d]
            global_t_feats = self.clip_model.encode_image(batch_global_t_inputs)    # [b, d]
            text_features = self.clip_model.encode_text(self.text_inputs)   # [c+1, d]
        global_rgb_feats /= global_rgb_feats.norm(dim=-1, keepdim=True)
        global_t_feats /= global_t_feats.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        global_rgb_similarity = (100.0 * global_rgb_feats @ text_features.T).softmax(dim=-1)    # [b, c+1]
        global_t_similarity = (100.0 * global_t_feats @ text_features.T).softmax(dim=-1)    # [b, c+1]
        global_similarity = torch.maximum(global_rgb_similarity, global_t_similarity) # [b, c+1]
        ## local
        (batch_patches_rgb_inputs, batch_patches_t_inputs), patch_shape = unfold_batch(batch_inputs, self.clip_model.visual.input_resolution)
        with torch.no_grad():
            local_rgb_feats, local_t_feats = self.clip_model.encode_image(batch_patches_rgb_inputs), self.clip_model.encode_image(batch_patches_t_inputs)
        local_rgb_feats /= local_rgb_feats.norm(dim=-1, keepdim=True)
        local_t_feats /= local_t_feats.norm(dim=-1, keepdim=True)

        local_rgb_similarity = (100.0 * local_rgb_feats @ text_features.T).softmax(dim=-1)    # [b*nBlock, c+1]
        local_t_similarity = (100.0 * local_t_feats @ text_features.T).softmax(dim=-1)    # [b*nBlock, c+1]
        rgb_similarity = local_rgb_similarity.view(patch_shape[0], patch_shape[1], self.num_class + len(self.extra_classes))    # [b, nBlock, c+1]
        t_similarity = local_t_similarity.view(patch_shape[0], patch_shape[1], self.num_class + len(self.extra_classes))    # [b, nBlock, c+1]
        local_similarity = torch.maximum(rgb_similarity, t_similarity)      # [b, nBlock, c+1]
        local_similarity = torch.max(local_similarity, 1)[0]    # [b, c+1]
        clip_scores = torch.maximum(local_similarity, global_similarity)       # [b, c+1]
        clip_scores = clip_scores[:, :self.num_class]   # [b, c]

        clip_logits = self.clip_fc(clip_scores.float()) + clip_scores.float()
        clip_loss = clip_train_loss(self.num_class, backbone_logits, backbone_masks, clip_logits, batch_data_samples, batch_inputs, self.roi_head.bbox_head.loss_cls)

        losses.update(clip_loss)

        return losses