from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MosaicQuickdrawDataset(BaseSegDataset):
    """Mosaic Quickdraw dataset.

    In segmentation map annotation for MosaicQuickdraw, 0 stands for background, which
    is not included in 100 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.mat'.
    """
    METAINFO = dict(
        classes=[
                    'airplane', 'angel', 'ant', 'apple', 'backpack',
                    'banana', 'baseball bat', 'basket', 'bear', 'bed',
                    'bench', 'bicycle', 'bird', 'book', 'bread',
                    'bridge', 'broccoli', 'bus', 'bush', 'butterfly',
                    'cake', 'car', 'carrot', 'castle', 'cat',
                    'cell phone', 'chair', 'church', 'clock', 'cloud',
                    'couch', 'cow', 'cup', 'dog', 'donut',
                    'duck', 'elephant', 'fence', 'fire hydrant', 'fish',
                    'flower', 'flying saucer', 'fork', 'frog', 'giraffe',
                    'grass', 'helicopter', 'horse', 'hospital', 'hot dog',
                    'house', 'kangaroo', 'keyboard', 'knife', 'laptop',
                    'leaf', 'light bulb', 'lightning', 'lion', 'mailbox',
                    'microwave', 'moon', 'motorbike', 'mountain', 'oven',
                    'palm tree', 'pig', 'pizza', 'rabbit', 'rainbow',
                    'remote control', 'sandwich', 'scissors', 'sheep', 'sink',
                    'skyscraper', 'snake', 'spoon', 'star', 'stop sign',
                    'streetlight', 'suitcase', 'sun', 'table', 'teddy-bear',
                    'television', 'tennis racquet', 'toaster', 'toilet', 'toothbrush',
                    'traffic light', 'train', 'tree', 'truck', 'umbrella',
                    'vase', 'windmill', 'wine bottle', 'wine glass', 'zebra'
                ],

        palette=[
                    (0, 141, 141), (0, 141, 94), (141, 141, 47), (141, 188, 0), (141, 94, 47),
                    (47, 47, 47), (47, 47, 0), (141, 0, 47), (0, 47, 47), (47, 94, 188),
                    (141, 141, 0), (0, 188, 0), (47, 0, 47), (94, 141, 94), (141, 0, 188),
                    (141, 141, 94), (94, 0, 94), (0, 141, 188), (141, 0, 0), (141, 94, 188),
                    (141, 94, 0), (0, 94, 188), (47, 188, 0), (188, 0, 94), (188, 47, 141),
                    (47, 94, 0), (47, 0, 141), (188, 188, 94), (0, 47, 141), (47, 0, 188),
                    (0, 188, 94), (0, 141, 47), (47, 141, 141), (188, 141, 188), (141, 94, 141),
                    (188, 47, 47), (188, 141, 94), (0, 47, 94), (94, 188, 47), (0, 94, 0),
                    (94, 141, 47), (94, 188, 188), (47, 47, 188), (0, 0, 188), (94, 47, 141),
                    (0, 188, 47), (94, 47, 188), (94, 188, 0), (188, 94, 47), (188, 188, 188),
                    (47, 0, 94), (94, 47, 0), (141, 0, 94), (47, 188, 188), (188, 188, 0),
                    (188, 0, 47), (188, 0, 141), (0, 47, 188), (94, 94, 188), (141, 141, 141),
                    (94, 141, 188), (94, 0, 47), (47, 47, 141), (47, 0, 0), (94, 47, 47),
                    (0, 141, 0), (141, 188, 141), (141, 47, 141), (141, 141, 188), (94, 94, 47),
                    (94, 141, 0), (0, 94, 141), (141, 47, 0), (0, 47, 0), (94, 188, 94),
                    (188, 94, 141), (188, 188, 141), (0, 188, 141), (188, 47, 188), (188, 0, 188),
                    (141, 0, 141), (188, 47, 94), (47, 188, 141), (0, 0, 94), (141, 94, 94),
                    (94, 94, 141), (141, 188, 188), (0, 0, 141), (47, 141, 94), (47, 47, 94),
                    (47, 141, 47), (47, 188, 94), (188, 94, 94), (94, 141, 141), (188, 0, 0),
                    (47, 141, 188), (141, 188, 47), (0, 0, 0), (188, 141, 0), (188, 188, 47)
                ]
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.mat',
            reduce_zero_label=True,
            **kwargs)
