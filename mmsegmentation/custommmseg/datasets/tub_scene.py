from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class TUBSceneDataset(BaseSegDataset):
    """TUBScene dataset.

    In segmentation map annotation for TUBScene, 0 stands for background, which
    is not included in 46 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.mat'.
    """
    METAINFO = dict(
        classes=[
                    'airplane', 'apple', 'banana', 'basket', 'bee',
                    'bench', 'bicycle', 'bird', 'bus', 'butterfly',
                    'car', 'cat', 'chair', 'cloud', 'couch',
                    'cow', 'cup', 'dog', 'duck', 'flower with stem',
                    'grapes', 'horse', 'house', 'moon', 'person walking',
                    'pig', 'rabbit', 'sheep', 'streetlight', 'sun',
                    'table', 'tree', 'truck', 'umbrella', 'wine bottle'
        ],

        palette=[
                    (0, 0, 255), (255, 0, 0), (255, 26, 185), (255, 211, 0), (0, 132, 246),
                    (0, 141, 70), (0, 62, 53), (167, 97, 62), (79, 0, 106), (0, 255, 246),
                    (62, 123, 141), (237, 167, 255), (211, 255, 149), (229, 26, 88), (0, 62, 53),
                    (132, 132, 0), (0, 62, 53), (0, 255, 149), (97, 0, 44), (202, 255, 0),
                    (44, 62, 0), (255, 202, 132), (0, 44, 97), (158, 114, 141), (158, 193, 255),
                    (255, 123, 176), (158, 9, 0), (132, 97, 202), (132, 220, 167), (255, 0, 246),
                    (0, 211, 255), (132, 255, 220), (88, 62, 53), (0, 62, 53), (0, 62, 53)
        ]
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='drawing.png',
            seg_map_suffix='class.mat',
            reduce_zero_label=True,
            **kwargs)
