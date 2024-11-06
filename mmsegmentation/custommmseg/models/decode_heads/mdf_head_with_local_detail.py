import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_ratio=0.):
        super(MLPBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MDFBlock(nn.Module):
    def __init__(self, mlp_seq: str='cwh', height_dim: int=0, width_dim: int=0, channel_dim: int=0, expension_ratio: int=4):
        super(MDFBlock, self).__init__()
        mlp_seq = mlp_seq.lower()
        assert set(mlp_seq).issubset({'c', 'h', 'w'})
        assert len(mlp_seq) < 4 and len(mlp_seq) == len(set(mlp_seq))
        self.mlp_seq = mlp_seq
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.channel_dim = channel_dim
        self.expension_ratio = expension_ratio

        if 'c' in self.mlp_seq:
            self.layer_norm_1 = nn.LayerNorm(channel_dim)
            self.mlp_block_channel = MLPBlock(channel_dim, channel_dim * expension_ratio, dropout_ratio=0.5)
        if 'w' in self.mlp_seq:
            self.layer_norm_2 = nn.LayerNorm(channel_dim)
            self.mlp_block_width = MLPBlock(width_dim, width_dim * expension_ratio, dropout_ratio=0.5)
        if 'h' in self.mlp_seq:
            self.layer_norm_3 = nn.LayerNorm(channel_dim)
            self.mlp_block_height = MLPBlock(height_dim, height_dim * expension_ratio, dropout_ratio=0.5)

    def _mlp_block_channel_forward(self, x):
        # x: (N, C, H, W)
        out = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        out = self.layer_norm_1(out)
        out = self.mlp_block_channel(out)
        out = out.permute(0, 3, 1, 2)  # (N, C, H, W)
        return out

    def _mlp_block_width_forward(self, x):
        # x: (N, C, H, W)
        out = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        out = self.layer_norm_2(out)
        out = out.permute(0, 3, 1, 2)  # (N, C, H, W)
        out = self.mlp_block_width(out)
        return out

    def _mlp_block_height_forward(self, x):
        # x: (N, C, H, W)
        out = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        out = self.layer_norm_3(out)
        out = out.permute(0, 3, 2, 1)  # (N, C, W, H)
        out = self.mlp_block_height(out)
        out = out.permute(0, 1, 3, 2)  # (N, C, H, W)
        return out

    def _mlp_block_forward(self, x, mode: str='c'):
        if mode == 'c':
            return self._mlp_block_channel_forward(x)
        elif mode == 'h':
            return self._mlp_block_height_forward(x)
        elif mode == 'w':
            return self._mlp_block_width_forward(x)

    def forward(self,x):
        # x: (N, C, H, W)
        out = x
        for mode in self.mlp_seq:
            shortcut = out
            out = self._mlp_block_forward(out, mode)
            out = out + shortcut

        return out


@MODELS.register_module()
class MDFHeadWithLocalDetail(BaseDecodeHead):
    """The MDF head with local detail.
    """

    def __init__(self, image_size, fusion_channels=0, interpolate_mode='bilinear', with_local_detail=False, mdf_block_num=0, mlp_seq='cwh', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.image_size = image_size
        self.fusion_channels = fusion_channels if fusion_channels>0 else self.channels
        self.interpolate_mode = interpolate_mode
        self.with_local_detail = with_local_detail
        self.mdf_block_num = mdf_block_num
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        if with_local_detail:
            self.stem_conv = ConvModule(
                    in_channels=self.in_channels[0],
                    out_channels=self.fusion_channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.fusion_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        if self.mdf_block_num > 0:
            self.mixer = nn.Sequential()
            for i in range(mdf_block_num):
                self.mixer.append(MDFBlock(mlp_seq, math.ceil(image_size / 4), math.ceil(image_size / 4), self.channels, 4))

        if self.fusion_channels != self.channels:
            self.channel_reduce_conv = ConvModule(
                in_channels=self.fusion_channels,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []

        if self.with_local_detail:
            stem_layer = inputs.pop(0)
            stem_layer = self.stem_conv(stem_layer)

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        if self.with_local_detail:
            out = out + stem_layer

        if self.fusion_channels != self.channels:
            out = self.channel_reduce_conv(out)

        if self.mdf_block_num > 0:
            out = self.mixer(out)

        out = self.cls_seg(out)

        return out
