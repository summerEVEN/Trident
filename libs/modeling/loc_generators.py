import torch
from torch import nn
from torch.nn import functional as F

from .models import register_generator


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


@register_generator('point')
class PointGenerator(nn.Module):
    """
        A generator for temporal "points"

        max_seq_len can be much larger than the actual seq length
    """

    def __init__(
            self,
            max_seq_len,  # max sequence length that the generator will buffer
            fpn_levels,  # number of fpn levels
            scale_factor,  # scale factor between two fpn levels
            regression_range,  # regression range (on feature grids)
            strides,  # stride of fpn levels
            use_offset=False  # if to align the points at grid centers
    ):
        super().__init__()
        # sanity check, # fpn levels and length divisible
        assert len(regression_range) == fpn_levels
        assert max_seq_len % scale_factor ** (fpn_levels - 1) == 0

        # save params
        self.max_seq_len = max_seq_len
        self.fpn_levels = fpn_levels
        self.scale_factor = scale_factor
        self.regression_range = regression_range
        self.strides = strides
        self.use_offset = use_offset

        # generate all points and buffer the list
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # loop over all points at each pyramid level
        for l in range(self.fpn_levels):
            stride = self.strides[l]
            reg_range = torch.as_tensor(self.regression_range[l], dtype=torch.float)
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)
            points = torch.arange(0, self.max_seq_len, stride)[:, None]
            # 生成一个从0到 self.max_seq_len 之间，以步幅 stride 递增的序列。这个序列表示了当前层级的时间“点”。
            # add offset if necessary (not in our current model)
            # use_offset 参数一直都是默认值,不使用
            if self.use_offset:
                points += 0.5 * stride
            # pad the time stamp with additional regression range / stride
            reg_range = reg_range[None].repeat(points.shape[0], 1)
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)
            # size: T x 4 (ts, reg_range, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))
            # 每个时间“点”列表包含了时间戳、回归范围和步幅等信息。

        return BufferList(points_list)

    def forward(self, feats):
        # feats will be a list of torch tensors
        assert len(feats) == self.fpn_levels
        pts_list = []
        feat_lens = [feat.shape[-1] for feat in feats]  # [2304, 1152, 576, 288, 144, 72]
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            assert feat_len <= buffer_pts.shape[0], "Reached max buffer length for point generator"
            # buffer_pts.shape : torch.Size([13824, 4])
            pts = buffer_pts[:feat_len, :]
            pts_list.append(pts)
        return pts_list
    
    """
    综上所述，PointGenerator 类实现了一个灵活的时间“点”生成器，能够根据给定的参数生成不同FPN层级的时间“点”，并与输入特征进行对齐，以用于后续的处理和分析。
    """
