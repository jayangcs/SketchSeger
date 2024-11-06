import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CustomRerange(BaseTransform):
    """Custom rerange the image pixel value.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results: dict) -> dict:
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        if img_min_value < img_max_value:
            # rerange to [0, 1]
            img = (img - img_min_value) / (img_max_value - img_min_value)
            # rerange to [min_value, max_value]
            img = img * (self.max_value - self.min_value) + self.min_value
            results['img'] = img
        else:
            img = img / img
            results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str
