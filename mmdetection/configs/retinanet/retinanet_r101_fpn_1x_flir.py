_base_ = './retinanet_r50_fpn_1x_flir.py'
load_from='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
