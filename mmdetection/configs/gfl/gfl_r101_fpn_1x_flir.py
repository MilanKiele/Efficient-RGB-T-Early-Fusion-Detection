_base_ = './gfl_r50_fpn_1x_flir.py'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
model = dict(
    type='GFLMediumFusion',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

# val_evaluator = dict(_delete_=True, type='VOCMetric', metric='mAP', eval_mode='11points')
# test_evaluator = val_evaluator
