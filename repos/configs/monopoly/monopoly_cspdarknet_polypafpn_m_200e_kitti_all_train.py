_base_ = [
    '_base_data_kitti_all.py',
    '_base_arch_cspdarknet_polypafpn.py',
    '_base_core_cyclic_200e.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py',
]

# mount dir
mount = True
mount_dir = 'repos/mmdet3d_mount/'

# reset cfg
model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192),
)

# extra env
find_unused_parameters = True
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]

load_from = 'ckpts/yolox_m_mmdet.pth'