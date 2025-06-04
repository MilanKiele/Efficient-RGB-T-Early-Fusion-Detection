_base_ = './emeDet_softmaxCECls_gfl_r50_fpn_1x_m3fd_kdMed2Ear.py'
model = dict(
    focusnet_config=dict(
        head=dict(
            loss=dict(
                type='FocusPickingLoss', use_sigmoid=False, loss_weight=1.0),
        )
    )
)