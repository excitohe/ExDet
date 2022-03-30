_base_ = ['monocon_dla34_dlaup_200e_kitti_all_train.py']

mount = True
mount_dir = 'repos/mmdet3d_mount/'

find_unused_parameters = True
model = dict(bbox_head=dict(type='MonoConHeadInference'))
