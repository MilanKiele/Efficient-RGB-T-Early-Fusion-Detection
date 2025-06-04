_base_ = './gfl_r50_fpn_1x_m3fd_kdMed2Ear.py'

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 33],
        gamma=0.1)
]

# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
