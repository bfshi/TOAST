#!/usr/bin/env python3

"""
Convnext-related models:
"imagenet_sup_rnx_tiny",
"imagenet_sup_rnx_small",
"imagenet_sup_rnx_base",
"imagenet22k_sup_rnx_base",
"imagenet22k_sup_rnx_large",
"imagenet22k_sup_rnx_xlarge",
"""
import torch
import torch.nn as nn
import torchvision as tv

from collections import OrderedDict
from timm.models.layers import trunc_normal_

from .convnext_backbone.convnext_bottom_up import convnext_bottom_up_base
from .convnext_backbone.convnext_top_down import convnext_top_down_base
from ..utils import logging
from .mlp import MLP

import loralib as lora
logger = logging.get_logger("visual_prompt")

class ConvNeXt_Top_Down(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ConvNeXt_Top_Down, self).__init__()
        self.cfg = cfg
        self.build_backbone(cfg, load_pretrain, vis=vis)
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                     [self.cfg.DATA.NUMBER_CLASSES],  # noqa
            special_bias=True
        )
        self.side = None

    def build_backbone(self, cfg, load_pretrain, vis):
        self.enc, self.feat_dim = convnext_top_down_base(pretrained=load_pretrain, cfg=cfg)

        if cfg.MODEL.TRANSFER_TYPE == 'toast':
            for k, p in self.enc.named_parameters():
                if "decoders" not in k and "prompt" not in k and "head" not in k:
                    p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    def forward(self, x, return_feature=False):
        x = self.enc(x)
        x = self.head(x)
        return x, None


class ConvNeXt_Bottom_Up(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ConvNeXt_Bottom_Up, self).__init__()
        self.cfg = cfg
        self.build_backbone(cfg, load_pretrain, vis=vis)
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                     [self.cfg.DATA.NUMBER_CLASSES],  # noqa
            special_bias=True
        )
        self.side = None

    def build_backbone(self, cfg, load_pretrain, vis):
        self.enc, self.feat_dim = convnext_bottom_up_base(pretrained=load_pretrain, cfg=cfg)

        if cfg.MODEL.TRANSFER_TYPE == 'linear':
            for k, p in self.enc.named_parameters():
                p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'lora':
            for n, m in list(self.enc.named_modules()):
                if isinstance(m, nn.Conv2d):
                    old_weight = m.weight.data
                    old_bias = m.bias.data
                    ks = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
                    stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
                    padding = m.padding[0] if isinstance(m.padding, tuple) else m.padding
                    setattr(self.enc, n, lora.Conv2d(m.in_channels, m.out_channels, kernel_size=ks,
                                                     stride=stride, groups=m.groups, padding=padding, r=4))
                    setattr(self.enc, n + '.weight.data', old_weight)
                    setattr(self.enc, n + '.bias.data', old_bias)
            lora.mark_only_lora_as_trainable(self.enc)
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    def forward(self, x, return_feature=False):
        x = self.enc(x)
        x = self.head(x)
        return x, None


