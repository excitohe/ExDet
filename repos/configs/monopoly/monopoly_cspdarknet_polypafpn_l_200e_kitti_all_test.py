_base_ = ['monopoly_cspdarknet_polypafpn_l_200e_kitti_all_train.py']

# mount dir
mount = True
mount_dir = 'repos/mmdet3d_mount/'

# extra env
find_unused_parameters = True
model = dict(bbox_head=dict(type='MonoConHeadInference'))
