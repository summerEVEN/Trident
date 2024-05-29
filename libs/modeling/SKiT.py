"""
这里是搭建后面的model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class SKiT(nn.Module):
    def __init__(self, args):
        super(SKiT, self).__init__()
        self.window_size = args.window_size
        # 初始化全为0的一个tensor，或者是一个全为负无穷的一个tensor
        # ！！！！！！！！！！！！ 这个细节还得进一步优化一下
        self.shape = (args.global_feature_length, )
        self.global_feature = torch.full(self.shape, float('-inf'))
        self.global_feature.requires_grad = False
        

        # 特征降维
        self.Linear0 = nn.Linear(args.vit_feature_length, args.local_feature_length)

        # 最后的预测分类头
        # 问问GPT，特征到预测分类怎么搞
        # Linear1: 使用融合的特征预测最后分类概率，用于准确率，交叉熵损失值的计算
        # Linear2：使用局部特征生成预测概率图，用于另外一个损失值的计算
        self.Linear1 = nn.Linear(args.local_feature_length, args.num_classes)
        self.Linear2 = nn.Linear(args.local_feature_length, args.num_classes)
        self.Linear3 = nn.Linear(args.local_feature_length, args.global_feature_length)
        self.Linear4 = nn.Linear(args.global_feature_length, args.local_feature_length)
        self.Linear5 = nn.Linear(args.local_feature_length, args.local_feature_length)

        self.dropout0 = nn.Dropout(p=0.2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # BatchNorm1d 输入的数据结构：
        # （N，C) 或者 （N，C，L）
        # N: batch_size, C:  number of features or channels, L: sequence length
        # 按照这个意思
        self.BN0 = nn.BatchNorm1d(64)
        self.BN1 = nn.BatchNorm1d(args.local_feature_length)
        self.BN2 = nn.BatchNorm1d(args.local_feature_length)

        self.l_aggregator1 = L_aggregator(args)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.conv1x1_1 = nn.Conv1d(768, 512, kernel_size=1)
        self.conv1x1_2 = nn.Conv1d(512, 512, kernel_size=1)


    def forward(self, frame, video_first, device):
        """
        frame: [100, 768]
        这是一个窗口的特征，桀桀桀

        下面描述一下具体的变化：
        1. 降维 [100, 768] -> [100, 512]
        2. L_aggregator 处理，变成局部特征：[100, 512] -> [100, 512]
        3. keypooling 处理，生成全局特征：[100, 512] -> [100, 64]
        4. fusion_head 融合： [100, 512] + [100, 64] -> [100, 512]
        5. 预测分类头: [100, 512] -> [100, 14]
        """
        # 初始化全局特征
        if video_first == True:
            self.global_feature = torch.full(self.shape, float('-inf'))
            self.global_feature = self.global_feature.to(device)

        # 特征降维
        frame= frame.unsqueeze(-1) 
        frame = frame.squeeze()
        frame = self.Linear0(frame) # 768 -> 512
        # frame = self.BN1(frame)
        # frame = self.BN1(self.relu1(frame))

        # frame= frame.unsqueeze(-1)
        # frame = self.conv1x1_2(frame)
        # frame = frame.squeeze()
        
        # 直接使用1*1卷积进行降维操作
        # frame= frame.unsqueeze(-1)
        # frame = self.conv1x1_1(frame)
        # frame = frame.squeeze()


        frame = frame.unsqueeze(0)
        # frame = self.l_aggregator1(frame, device)
        output = self.l_aggregator1(frame, device)
        output = output.squeeze(0)

        # output = self.BN1(self.relu1(output))
        # output = self.Linear0(output)
        local_features = self.Linear3(output) # 512 -> 64
        # local_features = self.BN0(local_features)

        x = torch.maximum(local_features[0], self.global_feature)
        output_sequence = x.clone().unsqueeze(0)
        for t in range(1, local_features.shape[0]):
            x = torch.maximum(local_features[t], x)
            output_sequence = torch.cat((output_sequence, x.unsqueeze(0)), dim = 0)

            self.global_feature = output_sequence[-1]
        # y=ReLU(BN(Conv(x)))+x

        """
        output_sequence.shape
        torch.Size([100, 64])
        output.shape
        torch.Size([100, 512])
        """
        # output1 = self.relu0(self.BN0(self.Linear4(output_sequence)) + output)
        # output1 = self.relu1(self.BN1(self.Linear5(output1)) + output1)

        output1 = self.Linear4(output_sequence) + output
        output1 = self.Linear5(output1) + output1

        output1 = self.Linear1(output1)
        output1 = self.softmax1(output1)

        output2 = self.Linear2(output)
        # output2 = self.softmax2(output2)

        return output1, output2
        

class L_aggregator(nn.Module):
    def __init__(self, args):
        super(L_aggregator, self).__init__()
        self.encoder_MSA1 = torch.nn.MultiheadAttention(args.aggregator_embed_dim, 
                                                args.aggregator_num_heads, 
                                                # dropout=0, 
                                                # bias=True, 
                                                # add_bias_kv=False, 
                                                # add_zero_attn=False, 
                                                # kdim=None, vdim=None, 
                                                batch_first=False, 
                                                # device=None, 
                                                # dtype=None
                                                )

        encoder_layer =  torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        # self.encoder_MSA2 = copy.deepcopy(self.encoder_MSA11)
        # self.decoder_MSA1 = copy.deepcopy(self.encoder_MSA1)
        # self.decoder_MSA2 = copy.deepcopy(self.encoder_MSA1)
        # self.cross_MSA1 = copy.deepcopy(self.encoder_MSA1)
        # self.cross_MSA2 = copy.deepcopy(self.encoder_MSA1)
        
    def forward(self, frame, device):
        """
        这个需要实现的功能是：
        1. 先把frame复制成两份
        2. 一份输入到MSA
        3. 一份MSA和cross MSA交叉
        具体细节需要进一步确定

        ！！！！！！！！！！！！
        这里只是attention，不是完整的transformer block

        print("frame.shape(): ", frame.shape()) [100, 512]
        """
        # frame = frame.unsqueeze(0)
        frame_clone = frame.clone()

        memory = self.transformer_encoder(frame)
        out = self.transformer_decoder(frame_clone, memory)


        # frame, wt = self.encoder_MSA11(frame, frame, frame)
        # frame, wt = self.encoder_MSA2(frame, frame, frame)
        # frame = self.encoder_MSA11(frame)
        # frame = self.encoder_MSA2(frame)
        # output, wt = self.decoder_MSA1(frame_clone, frame_clone, frame_clone)
        # output, wt = self.cross_MSA1(frame, output, output)
        # output, wt = self.decoder_MSA2(output, output, output)
        # output, wt = self.cross_MSA2(frame, output, output)
        # output = output.squeeze(0)


        return out

class key_pooling(nn.Module):
    def __init__(self, args):
        super(key_pooling, self).__init__()
        self.Linear1 = nn.Linear(args.local_feature_length, args.global_feature_length)
        # self.shape = (args.global_feature_length, )
        # self.global_feature = torch.full(self.shape, float('-inf'))

    def forward(self, local_features, global_feature):
        """
        这个部分的具体实现功能的理解存在进一步提升的空间。
        Args:
            local_features: args.window_size个局部特征 经过线性变换后的张量  
                [args.window_size, local_feature.length] 
                [100, 512]
            global_feature: 仅仅只是当前最新的全局信息 
                [1, global_feature.length]
                [1, 64]
        return:
            output_sequence: 这100个local_features对应的全局特征 [100, 64]
        """
        # 这一行代码是指，当输入的是一个新的视频的时候，需要把全局变量重置

        local_features = self.Linear1(local_features)

        # output_sequence = global_feature.unsqueeze(0).expand_as(local_features)
        # expanded_l.cuda()
        # output_sequence = torch.maximum(local_features, expanded_l)

        if local_features.numel() > 0:
            x = torch.maximum(local_features[0], global_feature)
            output_sequence = x.clone().unsqueeze(0)
            for t in range(1, local_features.shape[0]):
                x = torch.maximum(local_features[t], x)
                output_sequence = torch.cat((output_sequence, x.unsqueeze(0)), dim = 0)

            global_feature_new = output_sequence[-1]

            return output_sequence, global_feature_new
        else:
            raise ValueError("local_features is NULL!!!!!!!!!!!!????????????????????????????")
            return None

class fusion_head(nn.Module):
    def __init__(self, args):
        super(fusion_head, self).__init__()
        self.Linear1 = nn.Linear(args.global_feature_length, args.local_feature_length)
        self.Linear2 = nn.Linear(args.local_feature_length, args.local_feature_length)

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        # 这个Dropout还有1d，2d，3d，问问GPT


    def forward(self, local_features, global_features):
        """
        融合局部特征和全局特征
        局部特征是 L-aggregator 一次 forward 输出的信息 [100, 512]
        全局特征是 key_pooling 一次 forward 输出的信息 [100, 64]
        """
        global_features = self.Linear1(global_features)
        # global_features = self.relu1(global_features)
        output = global_features + local_features
        output = self.Linear2(output) + output
        # output = self.relu2(self.Linear2(output)) + output

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _generate_square_subsequent_mask(ls, sz):
            mask = (torch.triu(torch.ones(ls, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask


# class Attention(nn.Module):
#     def __init__(self,
#                  dim,   # 输入token的dim
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)

#         self.q_linear = nn.Linear(dim, dim)
#         self.k_linear = nn.Linear(dim, dim)
#         self.v_linear = nn.Linear(dim, dim)

#     def forward(self, q, k, v):
#         # [batch_size, num_patches + 1, total_embed_dim]
#         if torch.equal(q, k) and torch.equal(k, v):
#             B, N, C = q.shape
#             x = q

#             # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
#             # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
#             # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#             q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#         elif torch.equal(k, v):
#             B, N, C = q.shape
#             q = self.q_linear(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)
#             k = self.k_linear(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)
#             v = self.v_linear(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)

#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x, 0
class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1  # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    # def forward(self, x, mask):
    def forward(self, x):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        # out_mask = F.interpolate(
        #     mask.to(x.dtype),
        #     size=torch.div(T, self.stride, rounding_mode='trunc'),
        #     mode='nearest'
        # ).detach()

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out

        # out = x * out_mask + self.drop_path_out(out)
        out = x + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        # return out, out_mask.bool()
        return out
    

class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
    
# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output
    
class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out