_base_ = [
    './gfl_r50_fpn_1x_coco.py'
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
dataset_type = 'M3FDDataset'
data_root = '/home/zx/rgbx-distillation/datasets/detection/'
model = dict(
    # type='GFL',
    type='GFLCLIP',
    backbone=dict(
        type='ResNetRGBTEarlyModifiedStem',
        in_channels=3,
        frozen_stages=-1),
    data_preprocessor=dict(
        _delete_=True,
        type='RGBTDetDataPreprocessor',
        thermal_mean=[84.1, 84.1, 84.1],  # thermal_mean
        thermal_std=[50.6, 50.6, 50.6],  # thermal_std
        rgb_mean=[128.2, 129.3, 125.3],   # rgb_mean
        rgb_std=[49.1, 50.2, 53.5],
        bgr_to_rgb=True,
        pad_size_divisor=32),  # rgb_std
    bbox_head=dict(num_classes=6),)

# dataset
backend_args = None

train_pipeline = [
    dict(type='LoadRGBTImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeRGBT', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlipRGBT', prob=0.5),
    dict(type='PackDetRGBTInputs')
]
test_pipeline = [
    dict(type='LoadRGBTImageFromFile', backend_args=backend_args),
    dict(type='ResizeRGBT', scale=(640, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetRGBTInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='M3FD_zxSceneSplit_coco_format/infrared_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='M3FD_zxSceneSplit_coco_format/infrared_test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'M3FD_zxSceneSplit_coco_format/infrared_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

