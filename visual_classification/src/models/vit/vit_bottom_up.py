import math
import logging
from functools import partial
from functools import reduce
from operator import mul
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_#,PatchEmbed
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from torch.nn.modules.utils import _pair
import numpy as np

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_visualization = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if return_visualization:
            attn_copy = attn[0].clone().detach().cpu().numpy()
            attn_copy = attn_copy[:, 0, 1:]
            # height, width = int(math.sqrt(attn_copy.shape[-1])), int(math.sqrt(attn_copy.shape[-1]))
            # attn_copy = attn_copy.reshape((attn_copy.shape[0], height, width))
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_visualization:
            return x, attn_copy
        return x, None



class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_visualization = False):
        x_attn, visualization_heads = self.attn(self.norm1(x), return_visualization = return_visualization)
        x = x + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, visualization_heads


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, config, prompt_config, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, truncate_embedding = "none"):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'

        self.prompt_config = prompt_config
        self.vit_config = config
        self.patch_size = patch_size

        if config.MODEL.TRANSFER_TYPE == 'prompt':
            print(self.prompt_config)
            num_tokens = self.prompt_config.NUM_TOKENS
            self.num_tokens = num_tokens
            self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
            if truncate_embedding == "none":
                self.truncate_embedding = depth - 1
            else:
                assert truncate_embedding <= depth - 1, f"number of prompt truncated is {truncate_embedding}, depth is {depth}, prompts truncated exceeds prompt depth"
                self.truncate_embedding = truncate_embedding

            if self.prompt_config.PROJECT > -1:
                # only for prepend / add
                prompt_dim = self.prompt_config.PROJECT
                self.prompt_proj = nn.Linear(
                    prompt_dim, config.hidden_size)
                nn.init.kaiming_normal_(
                    self.prompt_proj.weight, a=0, mode='fan_out')
            else:
                prompt_dim = embed_dim
                self.prompt_proj = nn.Identity()

            if self.prompt_config.INITIATION == "random":
                patch_size_tuple = _pair(patch_size)
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size_tuple, 1) + prompt_dim))  # noqa
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

                if self.prompt_config.DEEP:  # noqa
                    total_d_layer = depth - 1
                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            else:
                raise ValueError("Other initiation scheme is not supported")


        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, return_spatial_feature=False):
        visualizations = []
        if self.vit_config.RETURN_VISUALIZATION:
            visualizations.append(x[0].clone().detach().cpu().numpy().transpose((1, 2, 0)))
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        B, nt, fd = x.shape
        if self.vit_config.MODEL.TRANSFER_TYPE == 'prompt':
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)

            if self.prompt_config.DEEP:
                num_layers = len(self.blocks)
                for i in range(num_layers):
                    if i == 0:
                        hidden_states, visualization_heads = self.blocks[i](x)
                    else:
                        if i <= self.deep_prompt_embeddings.shape[0] and i > self.deep_prompt_embeddings.shape[0] - self.truncate_embedding and i < 1:
                            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                                self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                            hidden_states = torch.cat((
                                hidden_states[:, :1, :],
                                deep_prompt_emb,
                                hidden_states[:, (1+self.num_tokens):, :]
                            ), dim=1)
                        if i < len(self.blocks) - 1:
                            hidden_states, visualization_heads = self.blocks[i](hidden_states)
                        else: 
                            hidden_states, visualization_heads = self.blocks[i](hidden_states, return_visualization=self.vit_config.RETURN_VISUALIZATION)
                x = hidden_states
            else:
                for i, blk in enumerate(self.blocks):
                    if i < len(self.blocks) - 1:
                        x, visualization_heads = blk(x)
                    else:
                        x, visualization_heads = blk(x, return_visualization=self.vit_config.RETURN_VISUALIZATION)
            if self.vit_config.RETURN_VISUALIZATION:
                visualization_heads = visualization_heads[:, self.num_tokens:]
        else:
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x, visualization_heads = blk(x)
                else:
                    x, visualization_heads = blk(x, return_visualization=self.vit_config.RETURN_VISUALIZATION)

        if self.vit_config.RETURN_VISUALIZATION:
            height, width = int(math.sqrt(visualization_heads.shape[-1])), int(math.sqrt(visualization_heads.shape[-1]))
            visualization_heads = visualization_heads.reshape((visualization_heads.shape[0], height, width))
            visualization_heads = np.repeat(visualization_heads, self.patch_size, axis = -2)
            visualization_heads = np.repeat(visualization_heads, self.patch_size, axis = -1)
            visualization_heads = np.append(visualization_heads, np.expand_dims(np.mean(visualization_heads, axis = 0), axis = 0), axis = 0)
            visualizations.append(visualization_heads)
    
        if self.vit_config.RETURN_VISUALIZATION:
            if self.vit_config.MODEL.TRANSFER_TYPE == 'prompt':
                attention_copy = x[0, 1 + self.num_tokens:].clone().detach().cpu().numpy()
            else:
                attention_copy = x[0, 1:].clone().detach().cpu().numpy()
            attention_copy = np.linalg.norm(attention_copy, axis = -1)
            height, width = int(math.sqrt(attention_copy.shape[0])), int(math.sqrt(attention_copy.shape[0]))
            attention_copy = attention_copy.reshape((height, width, -1))
            attention_copy = np.repeat(attention_copy, self.patch_size, axis = 0)
            attention_copy = np.repeat(attention_copy, self.patch_size, axis = 1)
            visualizations.append(attention_copy)

        x = self.norm(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)

        if self.vit_config.RETURN_VISUALIZATION:
            return self.head(x), visualizations
        return self.head(x), None
        

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()



@register_model
def vit_tiny_patch16_224(pretrained=False, cfg = None, prompt_cfg = None, **kwargs):
    model = VisionTransformer(
        config=cfg, prompt_config=prompt_cfg,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, cfg = None, prompt_cfg = None, **kwargs):
    assert cfg is not None, "cfg cannot be None!"
    # assert prompt_cfg is not None, "prompt cfg cannot be None!"
    model = VisionTransformer(
        config = cfg, prompt_config = prompt_cfg, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes = -1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load(cfg.MODEL.MODEL_ROOT, map_location="cpu")
        model.load_state_dict(state_dict["model"] if "model" in state_dict.keys() else state_dict, strict=False)
    return model, model.embed_dim
    


@register_model
def vit_base_patch16_224(pretrained=False, cfg = None, prompt_cfg = None, **kwargs):
    model = VisionTransformer(
        config=cfg, prompt_config=prompt_cfg,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes = -1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load(cfg.MODEL.MODEL_ROOT, map_location="cpu")
        model.load_state_dict(state_dict["model"] if "model" in state_dict.keys() else state_dict, strict=False)
    return model, model.embed_dim


@register_model
def vit_large_patch16_224(pretrained=False, cfg = None, prompt_cfg = None, **kwargs):
    model = VisionTransformer(
        config=cfg, prompt_config=prompt_cfg,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, num_classes = -1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load(cfg.MODEL.MODEL_ROOT, map_location="cpu")
        model.load_state_dict(state_dict["model"] if "model" in state_dict.keys() else state_dict, strict=False)
    return model, model.embed_dim

