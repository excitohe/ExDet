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
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96)
)

# extra env
find_unused_parameters = True
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]

load_from = 'ckpts/yolox_tiny_mmdet.pth'
