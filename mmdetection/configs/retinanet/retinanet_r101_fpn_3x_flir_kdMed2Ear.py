_base_ = './retinanet_r50_fpn_3x_flir_kdMed2Ear.py'

load_from='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth'

model = dict(
    backbone=dict(
        type='ResNetRGBTEarlyModifiedStem',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))