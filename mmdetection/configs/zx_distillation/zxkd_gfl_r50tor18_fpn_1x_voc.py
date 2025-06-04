_base_ = [
    '../gfl/gfl_r18_fpn_1x_voc0712.py'
]
teacher_ckpt = '/home/zx/rgbx-distillation/code/mmdetection-main/runs_voc/gfl_r50_fpn_1x_bs2x8_voc0712/epoch_4.pth'  # noqa
model = dict(
    type='KDGFL',
    teacher_config='/home/zx/rgbx-distillation/code/mmdetection-main/configs/gfl/gfl_r50_fpn_1x_voc0712.py',
    teacher_ckpt=teacher_ckpt,
    eval_teacher=True,
    bbox_head=dict(
        type='KDGFLHead',
        cls_kd_loss=dict(type='KnowledgeDistillationKLDivLoss',
                         reduction='mean',
                         loss_weight=0.,
                         T=10),
        box_kd_loss=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
    ))
