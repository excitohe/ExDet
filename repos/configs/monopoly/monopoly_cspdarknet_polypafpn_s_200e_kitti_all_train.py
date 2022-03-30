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
model = dict(bbox_head=dict(in_channels=128, feat_channels=128))

# extra env
find_unused_parameters = True
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]

load_from = 'ckpts/yolox_s_mmdet.pth'
