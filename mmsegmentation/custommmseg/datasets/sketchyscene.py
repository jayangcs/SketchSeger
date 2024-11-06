from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SketchySceneDataset(BaseSegDataset):
    """SketchyScene dataset.

    In segmentation map annotation for SketchyScene, 0 stands for background, which
    is not included in 46 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.mat'.
    """
    METAINFO = dict(
        classes=[
                    'airplane', 'apple', 'balloon', 'banana', 'basket',
                    'bee', 'bench', 'bicycle', 'bird', 'bottle',
                    'bucket', 'bus', 'butterfly', 'car', 'cat',
                    'chair', 'chicken', 'cloud', 'cow', 'cup',
                    'dinnerware', 'dog', 'duck', 'fence', 'flower',
                    'grape', 'grass', 'horse', 'house', 'moon',
                    'mountain', 'people', 'picnic rug', 'pig', 'rabbit',
                    'road', 'sheep', 'sofa', 'star', 'street lamp',
                    'sun', 'table', 'tree', 'truck', 'umbrella',
                    'others'
        ],

        palette=[
                    (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 26, 185), (255, 211, 0),
                    (0, 132, 246), (0, 141, 70), (0, 62, 53), (167, 97, 62), (0, 62, 53),
                    (0, 62, 53), (79, 0, 106), (0, 255, 246), (62, 123, 141), (237, 167, 255),
                    (211, 255, 149), (185, 79, 255), (229, 26, 88), (132, 132, 0), (0, 62, 53),
                    (0, 62, 53), (0, 255, 149), (97, 0, 44), (246, 132, 18), (202, 255, 0),
                    (44, 62, 0), (0, 53, 193), (255, 202, 132), (0, 44, 97), (158, 114, 141),
                    (79, 185, 18), (158, 193, 255), (149, 158, 123), (255, 123, 176), (158, 9, 0),
                    (255, 185, 185), (132, 97, 202), (0, 62, 53), (158, 0, 114), (132, 220, 167),
                    (255, 0, 246), (0, 211, 255), (132, 255, 220), (88, 62, 53), (0, 62, 53),
                    (0, 62, 53)
        ]
    )

    def __init__(self, reduce_zero_label=True, **kwargs) -> None:
        super().__init__(
            img_suffix='drawing.png',
            seg_map_suffix='class.mat',
            reduce_zero_label=reduce_zero_label,
            **kwargs)
