_base_ = [
    './gfl_r50_fpn_1x_m3fd.py'
]
data_root = '/home/zx/rgbx-distillation/datasets/detection/'
dataset_type = 'FLIRDataset'
model = dict(
    data_preprocessor=dict(
        type='RGBTDetDataPreprocessor',
        thermal_mean=[135.7, 135.7, 135.7],    # thermal_mean
        thermal_std=[63.6, 63.6, 63.6],    # thermal_std
        rgb_mean=[149.4, 148.7, 141.7],   # rgb_mean
        rgb_std=[49.3, 52.8, 59.0]),  # rgb_std
    bbox_head=dict(num_classes=3)
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='FLIR_coco/infrared_train.json',))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='FLIR_coco/infrared_test.json',))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'FLIR_coco/infrared_test.json',
    )
test_evaluator = val_evaluator