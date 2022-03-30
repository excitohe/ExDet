# Installation

## A from-scratch setup script

Here is a full script for setting up Exmo.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python setup.py develop
cd ..

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
python setup.py develop
cd ..

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc0
git checkout -b excito.1.0.dev
python setup.py develop
cd ..
```

## Dataset Preparation

It is recommended to symlink the dataset root to `$Exmo/data`.

```
Exmo
├── mmdetection
├── mmdetection3d
├── mmsegmentation
├── tools
├── repos
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne

```

### KITTI

Generate info files by running:
```
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```