_base_ = './retinanet_r50_fpn_1x_flir.py'

# teacher model: medium fusion retinanet res101: result: bbox_mAP_copypaste: 0.390 0.723 0.357 0.248 0.579 0.703
teacher_ckpt = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/FLIR_rgbtMedium_retinanet_r101_fpn_1x_bs4_coco3xPretrain/epoch_12.pth'
teacher_config = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/FLIR_rgbtMedium_retinanet_r101_fpn_1x_bs4_coco3xPretrain/20240223_133305/vis_data/config.py'
# model
model = dict(
    type='KDRetinaNetCLIP',
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt,
    eval_teacher=True,
    corekd_cfg=dict(type='MSELoss', loss_weight=1)
)