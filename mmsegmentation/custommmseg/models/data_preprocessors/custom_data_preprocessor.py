from typing import Any, Dict

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS


@MODELS.register_module()
class CustomDataPreProcessor(BaseDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        data['inputs'] = torch.as_tensor(data['inputs'], dtype=torch.float)

        return data
