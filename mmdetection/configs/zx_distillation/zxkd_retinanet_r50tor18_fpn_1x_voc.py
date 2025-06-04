_base_ = [
    './retinanet_r18_fpn_1x_voc.py'
]

teacher_ckpt = '/home/zx/rgbx-distillation/code/mmdetection-main/ckpts/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth'
# model
model = dict(
    type='KDRetinaNet',
    teacher_config='/home/zx/rgbx-distillation/code/mmdetection-main/configs/zx_distillation/retinanet_r50_fpn_1x_voc.py',
    teacher_ckpt=teacher_ckpt,
    eval_teacher=True,
    bbox_head=dict(
        type='KDRetinaHead',
        cls_kd_loss=dict(type='KnowledgeDistillationKLDivLoss',
                         reduction='mean',
                         loss_weight=10.0,
                         T=10),
        box_kd_loss=dict(loss_weight=1.0, type='GIoULoss'),
        # box_kd_loss=dict(loss_weight=1.0, type='CIoULoss'),
        ))