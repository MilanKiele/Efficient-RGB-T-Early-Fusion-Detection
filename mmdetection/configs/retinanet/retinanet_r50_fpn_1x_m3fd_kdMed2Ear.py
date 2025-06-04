_base_ = './retinanet_r50_fpn_1x_m3fd.py'

# teacher model: medium fusion retinanet res101: result: bbox_mAP_copypaste: 0.345 0.555 0.355 0.048 0.415 0.710
teacher_ckpt = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgbtMedium_retinanet_r101_fpn_1x_bs4_coco3xPretrain/epoch_10.pth'
teacher_config = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgbtMedium_retinanet_r101_fpn_1x_bs4_coco3xPretrain/20240223_132338/vis_data/config.py'
# model
model = dict(
    type='KDRetinaNetCLIP',
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt,
    eval_teacher=True,
    corekd_cfg=dict(type='MSELoss', loss_weight=1)
)