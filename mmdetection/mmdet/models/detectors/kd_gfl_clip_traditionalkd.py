# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .kd_gfl_clip import KDGFLCLIP
from typing import Union
from torch import Tensor
from mmdet.structures import SampleList
import torch


@MODELS.register_module()
class TraditionalKDGFLCLIP(KDGFLCLIP):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        losses = super().loss(batch_inputs, batch_data_samples)  # CLIP weakly supervised learning

        # knowledge distillation
        x = self.extract_feat(batch_inputs)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(batch_inputs)

        kd_losses = []
        for px, tx in zip(x, teacher_x):
            kd_losses.append(self.corekd_loss(px, tx))
        kd_losses = dict(kd_losses=kd_losses)

        losses.update(kd_losses)
        return losses
