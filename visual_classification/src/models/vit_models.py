#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models
import loralib as lora

from .mlp import MLP
from ..utils import logging
import sys
from .vit.vit_top_down import *
from .vit.vit_bottom_up import *
logger = logging.get_logger("visual_prompt")

class Topdown_ViTClass(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(Topdown_ViTClass, self).__init__()
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            print("prompt config loaded! ")
            prompt_cfg = cfg.MODEL.PROMPT
            print(prompt_cfg)
        else:
            prompt_cfg = None
        self.cfg = cfg
        self.build_backbone(cfg, prompt_cfg, load_pretrain, vis=vis)
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [self.cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )
        self.side = None
    
    def build_backbone(self, cfg, prompt_cfg, load_pretrain, vis):
        if cfg.MODEL.SIZE == 'base':
            self.enc, self.feat_dim = vit_topdown_base_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)
        elif cfg.MODEL.SIZE == 'large':
            self.enc, self.feat_dim = vit_topdown_large_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)

        if cfg.MODEL.TRANSFER_TYPE == 'toast':
            for k, p in self.enc.named_parameters():
                if "decoders" not in k and "prompt" not in k and "head" not in k:
                    p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'toast-lite':
            for decoder in self.enc.decoders:
                old_weight = decoder.linear.weight.data
                old_weight2 = decoder.linear2.weight.data
                decoder.linear = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4, bias=False)
                decoder.linear2 = lora.Linear(self.enc.embed_dim, self.enc.embed_dim, r=4, bias=False)
                decoder.linear.weight.data = old_weight
                decoder.linear2.weight.data = old_weight2
            lora.mark_only_lora_as_trainable(self.enc)
            self.enc.prompt.requires_grad = True
            self.enc.top_down_transform.requires_grad = True
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    def forward(self, x, return_feature=False):
        # print(self.enc(x))
        x, visualizations = self.enc(x) 
        # x = self.enc(x) 
        x = self.head(x)
        return x, visualizations


class Bottomup_ViTClass(nn.Module):
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(Bottomup_ViTClass, self).__init__()
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            print("prompt config loaded! ")
            prompt_cfg = cfg.MODEL.PROMPT
            print(prompt_cfg)
        else:
            prompt_cfg = None
        self.cfg = cfg
        self.build_backbone(cfg, prompt_cfg, load_pretrain, vis=vis)
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [self.cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )
        self.side = None
    
    def build_backbone(self, cfg, prompt_cfg, load_pretrain, vis):
        if cfg.MODEL.SIZE == 'base':
            self.enc, self.feat_dim = vit_base_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)
        elif cfg.MODEL.SIZE == 'large':
            self.enc, self.feat_dim = vit_large_patch16_224(pretrained=load_pretrain, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)

        if cfg.MODEL.TRANSFER_TYPE == 'linear':
            for k, p in self.enc.named_parameters():
                p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'prompt':
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "head" not in k:
                    p.requires_grad = False
        elif cfg.MODEL.TRANSFER_TYPE == 'lora':
            for block in self.enc.blocks:
                old_weight = block.attn.qkv.weight.data
                old_bias = block.attn.qkv.bias.data
                block.attn.qkv = lora.Linear(self.enc.embed_dim, self.enc.embed_dim*3, r=4)
                block.attn.qkv.weight.data = old_weight
                block.attn.qkv.bias.data = old_bias
            lora.mark_only_lora_as_trainable(self.enc)
        elif cfg.MODEL.TRANSFER_TYPE == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise NotImplementedError

    def forward(self, x, return_feature=False):
        x, visualizations = self.enc(x)  
        x = self.head(x)
        return x, visualizations

