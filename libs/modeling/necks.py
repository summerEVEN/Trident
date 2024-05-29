import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm


# 主打进行一个金字塔下采样网络的设计，里面使用了一个深度可分离卷积
@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i],
                scale_factor=self.scale_factor,
                mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x, _ = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )

        return fpn_feats, fpn_masks


# 目前所有的参数文件里面，使用的 neck 模块都是 identity
# 这是一个用于构建恒等映射（identity mapping）的模块
@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i + self.start_level] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # 通过这个源码，实际上输入的数据和输出的数据是 恒等映射 的关系
        # 最后视参数 with_ln 决定是否要进行层归一化
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        # 在模型打印的数据里面，就是一堆归一化的
        fpn_feats = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )

        return fpn_feats, fpn_masks
    
    # 在meta_archs.py 文件中，调用了这个函数，其中start_level和end_level没有赋值，进过初始化处理之后的值为0, len(out_channel)
    # 这里好像是对每一个瞬间的特征进行层归一化，如果with_ln为true

    # 每一层进行归一化，但是数值没有变化，一点都没有变化

"""
这一批的数据的大小是这样的：
I3D特征：[2048, 249]
249 是时间维度

video_list[0]["video_id"]
'video_test_0000004'
video_list[0]["feats"]
tensor([[0.3195, 0.2756, 0.2920,  ..., 0.1655, 0.1615, 0.1866],
        [0.8109, 0.7967, 0.7475,  ..., 0.3524, 0.3696, 0.4316],
        [0.2187, 0.2000, 0.2268,  ..., 0.2527, 0.2606, 0.2528],
        ...,
        [0.2884, 0.1338, 0.1279,  ..., 0.2350, 0.1478, 0.1486],
        [0.0469, 0.0793, 0.0603,  ..., 0.0244, 0.0228, 0.0234],
        [0.1757, 0.4676, 0.2828,  ..., 0.0602, 0.0646, 0.0359]],
       device='cuda:0')
video_list[0]["feats"].shape
torch.Size([2048, 249])
video_list[0]["segments"]
tensor([[ -0.5000,   6.2500],
        [ 83.5000,  89.5000],
        [137.5000, 154.0000],
        [210.2500, 220.7500],
        [  5.5000,   9.2500],
        [154.0000, 165.2500],
        [225.2500, 235.7500]], device='cuda:0')
video_list[0]["segments"].shape
torch.Size([7, 2])

inputs[0].shape
torch.Size([1, 512, 2304])
inputs[1].shape
torch.Size([1, 512, 1152])
inputs[2].shape
torch.Size([1, 512, 576])
inputs[3].shape
torch.Size([1, 512, 288])
inputs[4].shape
torch.Size([1, 512, 144])
inputs[5].shape
torch.Size([1, 512, 72])
len(fpn_masks)
6
fpn_masks[0]
tensor([[[ True,  True,  True,  ..., False, False, False]]], device='cuda:0')
fpn_masks[0].shape
torch.Size([1, 1, 2304])
fpn_masks[1].shape
torch.Size([1, 1, 1152])
fpn_masks[2].shape
torch.Size([1, 1, 576])
fpn_masks[3].shape
torch.Size([1, 1, 288])
fpn_masks[4].shape
torch.Size([1, 1, 144])
fpn_masks[5].shape
torch.Size([1, 1, 72])
fpn_masks[0][248] fpn_masks[0][249]

fpn_masks[0][0][0][248]
tensor(True, device='cuda:0')
fpn_masks[0][0][0][249]
tensor(False, device='cuda:0')
fpn_masks[1][0][0][124]
tensor(True, device='cuda:0')
fpn_masks[1][0][0][125]
tensor(False, device='cuda:0')
fpn_masks[2][0][0][62]
tensor(True, device='cuda:0')
fpn_masks[2][0][0][63]
tensor(False, device='cuda:0')
"""
    
