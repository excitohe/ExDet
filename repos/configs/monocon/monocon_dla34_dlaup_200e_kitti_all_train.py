_base_ = [
    '_base_dataset_kitti_all.py',
    '_base_model_dla34.py',
    '_base_schedule_cyclic_200e.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py',
]

mount = True
mount_dir = 'repos/mmdet3d_mount/'

find_unused_parameters = True
checkpoint_config = dict(interval=5)
workflow = [('train', 1)]
