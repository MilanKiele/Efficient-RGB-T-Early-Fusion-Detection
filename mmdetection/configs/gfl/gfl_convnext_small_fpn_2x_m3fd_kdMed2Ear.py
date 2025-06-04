_base_ = './gfl_convnext_fpn_1x_m3fd.py'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth'
# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

# teacher model: medium fusion gfl res101: result: bbox_mAP_copypaste: 0.372 0.591 0.383 0.096 0.431 0.717
teacher_ckpt = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgbtMedium_gfl_connext_small_fpn_2x_bs4/epoch_20.pth'
teacher_config = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_rgbtMedium_gfl_connext_small_fpn_2x_bs4/gfl_convnext_fpn_1x_m3fd.py'
# model
model = dict(
    type='KDGFLCLIP',
    teacher_config=teacher_config,
    teacher_ckpt=teacher_ckpt,
    eval_teacher=True,
    corekd_cfg=dict(type='MSELoss', loss_weight=1)
)