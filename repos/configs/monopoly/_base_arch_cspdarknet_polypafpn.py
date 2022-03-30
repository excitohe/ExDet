model = dict(
    type='MonoCon',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='POLYPAFPNV1',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1
    ),
    bbox_head=dict(
        type='MonoConHead',
        num_classes=3,
        in_channels=64,
        feat_channels=64,
        num_alpha_bins=12,
        loss_ctr2d_hmp=dict(type='MonoConGaussianFocalLoss', loss_weight=1.0),
        loss_ctr2d_res=dict(type='L1Loss', loss_weight=1.0),
        loss_kpt2d_hmp=dict(type='MonoConGaussianFocalLoss', loss_weight=1.0),
        loss_kpt2d_res=dict(type='L1Loss', loss_weight=1.0),
        loss_kpt2d_ofs=dict(type='L1Loss', loss_weight=1.0),
        loss_depth=dict(type='MonoConUncertaintyLoss', loss_weight=1.0),
        loss_dim2d=dict(type='L1Loss', loss_weight=0.1),
        loss_dim3d=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_alpha_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_alpha_reg=dict(type='L1Loss', loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4)
)
