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
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256)
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

# extra env
find_unused_parameters = True
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]

load_from = 'ckpts/yolox_l_mmdet.pth'
