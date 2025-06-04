_base_ = './gfl_r50_fpn_1x_m3fd_kdMed2Ear.py'
load_from = '/home/zx/rgbx-distillation/code/mmdetection-main/runs/train/M3FD_coreKD_rgbtEarly_zxModifiedStem_gfl_r101tor50_fpn_1x_bs4_clipTransfer/epoch_10.pth'
model = dict(
    type='KDGFLCLIPDetFocusNetCls',
    bbox_head=dict(
        type='GFLHeadzxReturnClsLogits',),
    focusnet_config=dict(
        backbone=dict(
            type='ResNetRGBTEarlyModifiedStemConv1S1',
            in_channels=3,
            depth=18,
            num_stages=4,
            out_indices=(3,),
            init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        ),
        neck=dict(type='CLSGlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=6,
            in_channels=512,
            loss=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            topk=(1, 5),
        )),
    rcnn_train_cfg=dict(pos_weight=-1, debug=False),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=64, sampling_ratio=0, pool_mode='max'),
        out_channels=3,
        featmap_strides=[1,]),
    assigner_config=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.2,
            neg_iou_thr=0.2,
            min_pos_iou=0.2,
            match_low_quality=False,
            ignore_iof_thr=-1),
    sampler_config=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
)

train_dataloader = dict(
    batch_size=2,)


# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=3,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=47,
        by_epoch=True,
        begin=3,
        end=50,
    )
]

# train, val, test setting
train_cfg = dict(max_epochs=50, val_interval=1)
# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)