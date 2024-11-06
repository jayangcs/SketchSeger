from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SKYSceneDataset(BaseSegDataset):
    """SKYScene dataset.

    In segmentation map annotation for SKYScene, 0 stands for background, which
    is not included in 46 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.mat'.
    """
    METAINFO = dict(
        classes=[
                    'airplane', 'apple', 'banana', 'bee', 'bench',
                    'bicycle', 'butterfly', 'cabin', 'car (sedan)', 'cat',
                    'chair', 'chicken', 'couch', 'cow', 'cup',
                    'dog', 'duck', 'flower', 'horse', 'pickup truck',
                    'pig', 'rabbit', 'sheep', 'songbird', 'strawberry',
                    'table', 'tree', 'umbrella', 'volcano', 'wine bottle'
        ],

        palette=[
                    (0, 0, 255), (255, 0, 0), (255, 26, 185), (0, 132, 246), (0, 141, 70),
                    (0, 62, 53), (0, 255, 246), (0, 44, 97), (62, 123, 141), (237, 167, 255),
                    (211, 255, 149), (185, 79, 255), (0, 62, 53), (132, 132, 0), (0, 62, 53),
                    (0, 255, 149), (97, 0, 44), (202, 255, 0), (255, 202, 132), (88, 62, 53),
                    (255, 123, 176), (158, 9, 0), (132, 97, 202), (167, 97, 62), (44, 62, 0),
                    (0, 211, 255), (132, 255, 220), (0, 62, 53), (79, 185, 18), (0, 62, 53)
        ]
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='drawing.png',
            seg_map_suffix='class.mat',
            reduce_zero_label=True,
            **kwargs)
