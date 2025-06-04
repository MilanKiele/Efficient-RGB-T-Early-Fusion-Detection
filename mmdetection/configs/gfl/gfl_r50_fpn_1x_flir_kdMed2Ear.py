_base_ = './gfl_r50_fpn_1x_flir.py'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth'

# teacher model: medium fusion gfl res101: result: bbox_mAP_copypaste: 0.440 0.772 0.429 0.326 0.601 0.764
teacher_ckpt = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/FLIR_rgbtMedium_gfl_r101_fpn_1x_bs4/epoch_10.pth'
teacher_config = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/FLIR_rgbtMedium_gfl_r101_fpn_1x_bs4/gfl_r101_fpn_1x_flir.py'
# model
model = dict(
    type='KDGFLCLIP',
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt,
    eval_teacher=True,
    corekd_cfg=dict(type='MSELoss', loss_weight=1.0)
)