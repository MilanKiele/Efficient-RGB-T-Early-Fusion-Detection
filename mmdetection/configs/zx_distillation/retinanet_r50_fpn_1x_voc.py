_base_ = [
    '../pascal_voc/retinanet_r50_fpn_1x_voc0712.py'
]

train_dataloader = dict(batch_size=2)


# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 4
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=4000),# batch_size = 2, so warmup = 500*16/2 = 4000
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]