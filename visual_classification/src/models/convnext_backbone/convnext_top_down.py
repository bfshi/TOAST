# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Decode_Block(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride):
        super().__init__()
        self.linear = nn.ConvTranspose2d(in_chans, out_chans, kernel_size, stride, bias=False)
        self.linear2 = nn.Conv2d(out_chans, out_chans, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        out = self.linear2(x)
        return x, out


class ConvNeXt_Top_Down(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=-1,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.decoders = nn.ModuleList()
        self.decoders.append(Decode_Block(dims[0], in_chans, kernel_size=4, stride=4))
        for i in range(3):
            self.decoders.append(Decode_Block(dims[i + 1], dims[i], kernel_size=2, stride=2))
        self.prompt = torch.nn.parameter.Parameter(torch.randn(dims[-1]), requires_grad=True)
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(dims[-1]), requires_grad=True)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def feedback(self, x):
        td = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            x, out = self.decoders[depth](x)
            td = [out] + td
        return td

    def forward_features(self, x, td=None):
        in_var = []
        out_var = []
        for i in range(4):
            in_var.append(x)
            if td is not None:
                x = x + td[i]
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_var.append(x)
        return x, in_var, out_var

    def forward(self, x, return_var_loss=False):
        input = x
        x, _, __ = self.forward_features(input)

        cos_sim = (F.normalize(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), dim=1) * F.normalize(
            self.prompt[None, ..., None, None], dim=1)).sum(dim=1, keepdim=True)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = (x.permute(0, 2, 3, 1) @ self.top_down_transform).permute(0, 3, 1, 2)
        td = self.feedback(x)

        x, in_var, out_var = self.forward_features(input, td)
        x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)

        var_loss = self.var_loss(in_var, out_var, x)

        return x

    def var_loss(self, in_var, out_var, x):
        recon_loss = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            recon, out = self.decoders[depth](out_var[depth].detach())
            target = in_var[depth].detach()
            recon_loss.append(F.mse_loss(recon, target))

        return 0.01 * sum(recon_loss)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register_model
def convnext_top_down_base(pretrained=False, cfg = None, **kwargs):
    model = ConvNeXt_Top_Down(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        state_dict = torch.load(cfg.MODEL.MODEL_ROOT, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=False)
    return model, 1024

