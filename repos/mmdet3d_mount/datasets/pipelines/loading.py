from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations3DMonoCon(LoadAnnotations3D):

    def __init__(self, with_kpts_2d=True, **kwargs):
        super().__init__(**kwargs)
        self.with_kpts_2d = with_kpts_2d

    def _load_kpts_2d(self, results):
        """ MonoCon special function to load 3D bounding box annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.
        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_kpts_2d'] = results['ann_info']['gt_kpts_2d']
        results['gt_kpts_valid_mask'] = results['ann_info']['gt_kpts_valid_mask'
                                                            ]
        return results

    def __call__(self, results):
        results = super().__call__(results)

        if self.with_kpts_2d:
            results = self._load_kpts_2d(results)

        return results