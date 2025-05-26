from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch as th # Menggunakan th sebagai alias untuk torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    avg_pool_nd, 
    zero_module, 
    normalization, # Ini adalah fungsi, bukan nn.Module, jadi tidak perlu L4Q
    timestep_embedding,
)
# SpatialTransformer akan diimpor dari ldm.modules.attention yang sudah dimodifikasi
from ldm.modules.attention import SpatialTransformer 

# --- Impor Fungsi Helper L4Q dari l4q_utils ---
try:
    # Menggunakan ..l4q karena openaimodel.py ada di ldm/modules/diffusionmodules/
    from ..l4q.l4q_utils import make_l4q_linear, make_l4q_conv2d
except ImportError:
    print("PERINGATAN KRITIKAL di openaimodel.py: Gagal mengimpor helper L4Q dari ..l4q.l4q_utils. Layer L4Q TIDAK AKAN BERFUNGSI.")
    # Fallback placeholder
    def make_l4q_linear(in_features, out_features, bias=True, l4q_enabled_passed=False, **kwargs):
        print(f"openaimodel.py: Fallback ke nn.Linear ({in_features}, {out_features}) karena impor helper L4Q gagal.")
        return nn.Linear(in_features, out_features, bias=bias)
    def make_l4q_conv2d(dims, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, l4q_enabled_passed=False, **kwargs):
        print(f"openaimodel.py: Fallback ke nn.Conv{dims}d ({in_channels}, {out_channels}) karena impor helper L4Q gagal.")
        if dims == 1: return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if dims == 2: return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if dims == 3: return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        raise ValueError(f"Dimensi tidak didukung: {dims}")

# dummy replace (tidak berubah)
def convert_module_to_f16(x): pass
def convert_module_to_f32(x): pass

class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spatial_dim: int, # Nama variabel konsisten
        embed_dim: int,
        num_heads_channels: int, # Ini adalah dim_head
        output_dim: int = None,
        l4q_params: dict = None,
    ):
        super().__init__()
        self.l4q_params = l4q_params
        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        self.positional_embedding = nn.Parameter(th.randn(spatial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        
        self.qkv_proj = make_l4q_conv2d(1, embed_dim, 3 * embed_dim, 1, l4q_enabled_passed=l4q_enabled,
                                        lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.c_proj = make_l4q_conv2d(1, embed_dim, output_dim or embed_dim, 1, l4q_enabled_passed=l4q_enabled,
                                      lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x): # x: B, C, H, W
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x.permute(0, 2, 1) 
        x = x + self.positional_embedding.to(x.dtype) 
        x = x.permute(0, 2, 1) 
        x = self.qkv_proj(x) 
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UpsampleOpenAI(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, l4q_params=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.l4q_params = l4q_params
        
        if use_conv:
            l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
            self.conv = make_l4q_conv2d(dims, self.channels, self.out_channels, 3, padding=padding, l4q_enabled_passed=l4q_enabled,
                                        lora_rank=self.l4q_params.get("lora_rank",4) if self.l4q_params else 4,
                                        n_bits=self.l4q_params.get("n_bits",4) if self.l4q_params else 4,
                                        alpha=self.l4q_params.get("alpha",1.0) if self.l4q_params else 1.0,
                                        quant_group_size=self.l4q_params.get("quant_group_size",-1) if self.l4q_params else -1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)
    def forward(self,x):
        return self.up(x)

class DownsampleOpenAI(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1, l4q_params=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.l4q_params = l4q_params
        stride = 2 if dims != 3 else (1, 2, 2)
        
        if use_conv:
            l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
            self.op = make_l4q_conv2d(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, l4q_enabled_passed=l4q_enabled,
                                      lora_rank=self.l4q_params.get("lora_rank",4) if self.l4q_params else 4,
                                      n_bits=self.l4q_params.get("n_bits",4) if self.l4q_params else 4,
                                      alpha=self.l4q_params.get("alpha",1.0) if self.l4q_params else 1.0,
                                      quant_group_size=self.l4q_params.get("quant_group_size",-1) if self.l4q_params else -1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(
        self, channels, emb_channels, dropout, out_channels=None, use_conv=False, 
        use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False, l4q_params=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv_skip = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        conv1 = make_l4q_conv2d(dims, channels, self.out_channels, 3, padding=1, l4q_enabled_passed=l4q_enabled,
                                lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv1)

        self.updown = up or down
        if up:
            self.h_upd = UpsampleOpenAI(channels, False, dims, l4q_params=None) 
            self.x_upd = UpsampleOpenAI(channels, False, dims, l4q_params=None)
        elif down:
            self.h_upd = DownsampleOpenAI(channels, False, dims, l4q_params=None)
            self.x_upd = DownsampleOpenAI(channels, False, dims, l4q_params=None)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        emb_linear_out_ch = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        emb_linear = make_l4q_linear(emb_channels, emb_linear_out_ch, l4q_enabled_passed=l4q_enabled,
                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        self.emb_layers = nn.Sequential(nn.SiLU(), emb_linear)
        
        out_conv = make_l4q_conv2d(dims, self.out_channels, self.out_channels, 3, padding=1, l4q_enabled_passed=l4q_enabled,
                                   lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(out_conv),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif self.use_conv_skip:
            self.skip_connection = make_l4q_conv2d(dims, channels, self.out_channels, 3, padding=1, l4q_enabled_passed=l4q_enabled,
                                                   lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.skip_connection = make_l4q_conv2d(dims, channels, self.out_channels, 1, l4q_enabled_passed=l4q_enabled,
                                                   lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(
        self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False,
        use_new_attention_order=False, l4q_params=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1: self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        self.qkv = make_l4q_conv2d(1, channels, channels * 3, 1, l4q_enabled_passed=l4q_enabled, # dims=1
                                   lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        if use_new_attention_order: self.attention = QKVAttention(self.num_heads)
        else: self.attention = QKVAttentionLegacy(self.num_heads)
        
        proj_out_layer = make_l4q_conv2d(1, channels, channels, 1, l4q_enabled_passed=l4q_enabled, # dims=1
                                         lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.proj_out = zero_module(proj_out_layer)
    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
    def _forward(self, x):
        b, c, *spatial = x.shape
        x_reshaped = x.reshape(b, c, -1)
        norm_x = self.norm(x_reshaped) # Normalisasi sebelum QKV
        qkv = self.qkv(norm_x) # qkv adalah conv1d
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_reshaped + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module): # Tidak ada L4Q di sini
    def __init__(self, n_heads): super().__init__(); self.n_heads = n_heads
    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
    @staticmethod
    def count_flops(model, _x, y): return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module): # Tidak ada L4Q di sini
    def __init__(self, n_heads): super().__init__(); self.n_heads = n_heads
    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)
    @staticmethod
    def count_flops(model, _x, y): return count_flops_attn(model, _x, y)

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape; num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops]); return

class UNetModel(nn.Module):
    def __init__(
        self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
        attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True,
        dims=2, num_classes=None, use_checkpoint=False, use_fp16=False,
        num_heads=-1, num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False,
        use_spatial_transformer=False, transformer_depth=1, context_dim=None,
        n_embed=None, legacy=True, l4q_params=None,
    ):
        super().__init__()
        if use_spatial_transformer: assert context_dim is not None
        if context_dim is not None:
            assert use_spatial_transformer
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig: context_dim = list(context_dim)
        if num_heads_upsample == -1: num_heads_upsample = num_heads # Tidak dipakai LDM
        if num_heads == -1: assert num_head_channels != -1
        if num_head_channels == -1: assert num_heads != -1

        self.image_size = image_size; self.in_channels = in_channels
        self.model_channels = model_channels; self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks; self.attention_resolutions = attention_resolutions
        self.dropout = dropout; self.channel_mult = channel_mult
        self.conv_resample = conv_resample; self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint; self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads; self.num_head_channels = num_head_channels
        self.predict_codebook_ids = n_embed is not None
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            make_l4q_linear(model_channels, time_embed_dim, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs),
            nn.SiLU(),
            make_l4q_linear(time_embed_dim, time_embed_dim, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs),
        )
        if self.num_classes is not None: self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(make_l4q_conv2d(dims, in_channels, model_channels, 3, padding=1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs))
        ])
        input_block_chans = [model_channels]; ch = model_channels; ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    curr_num_heads = num_heads if num_heads!=-1 else ch // num_head_channels if num_head_channels!=-1 else 8
                    curr_dim_head = num_head_channels if num_head_channels!=-1 else ch // curr_num_heads if curr_num_heads!=0 else 64
                    if curr_num_heads == 0 : curr_num_heads = ch // curr_dim_head # Safety
                    layers.append(
                        SpatialTransformer(ch, curr_num_heads, curr_dim_head, depth=transformer_depth, context_dim=context_dim, l4q_params=self.l4q_params)
                        if use_spatial_transformer else
                        AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=curr_num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order, l4q_params=self.l4q_params)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch_down = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch_down, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True, l4q_params=self.l4q_params)
                    if resblock_updown else
                    DownsampleOpenAI(ch, conv_resample, dims=dims, out_channels=out_ch_down, l4q_params=self.l4q_params)
                ))
                ch = out_ch_down; input_block_chans.append(ch); ds *= 2
        
        mid_num_heads = num_heads if num_heads!=-1 else ch // num_head_channels if num_head_channels!=-1 else 8
        mid_dim_head = num_head_channels if num_head_channels!=-1 else ch // mid_num_heads if mid_num_heads!=0 else 64
        if mid_num_heads == 0 : mid_num_heads = ch // mid_dim_head

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params),
            SpatialTransformer(ch, mid_num_heads, mid_dim_head, depth=transformer_depth, context_dim=context_dim, l4q_params=self.l4q_params)
            if use_spatial_transformer else
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=mid_num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order, l4q_params=self.l4q_params),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params),
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    curr_num_heads = num_heads if num_heads!=-1 else ch // num_head_channels if num_head_channels!=-1 else 8
                    curr_dim_head = num_head_channels if num_head_channels!=-1 else ch // curr_num_heads if curr_num_heads!=0 else 64
                    if curr_num_heads == 0 : curr_num_heads = ch // curr_dim_head
                    layers.append(
                        SpatialTransformer(ch, curr_num_heads, curr_dim_head, depth=transformer_depth, context_dim=context_dim, l4q_params=self.l4q_params)
                        if use_spatial_transformer else
                        AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=curr_num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order, l4q_params=self.l4q_params)
                    )
                if level and i == num_res_blocks:
                    out_ch_up = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch_up, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True, l4q_params=self.l4q_params)
                        if resblock_updown else
                        UpsampleOpenAI(ch, conv_resample, dims=dims, out_channels=out_ch_up, l4q_params=self.l4q_params)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            zero_module(make_l4q_conv2d(dims, model_channels, out_channels, 3, padding=1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                make_l4q_conv2d(dims, model_channels, n_embed, 1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs),
            )
    def convert_to_fp16(self): self.input_blocks.apply(convert_module_to_f16); self.middle_block.apply(convert_module_to_f16); self.output_blocks.apply(convert_module_to_f16)
    def convert_to_fp32(self): self.input_blocks.apply(convert_module_to_f32); self.middle_block.apply(convert_module_to_f32); self.output_blocks.apply(convert_module_to_f32)
    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        assert (y is not None) == (self.num_classes is not None)
        hs = []; 
        if timesteps is None: timesteps = th.zeros(x.shape[0], device=x.device) # Default timesteps if None
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None: emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for module in self.input_blocks: h = module(h, emb, context); hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks: h = th.cat([h, hs.pop()], dim=1); h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids: return self.id_predictor(h)
        else: return self.out(h)

class EncoderUNetModel(nn.Module): # UNet Encoder untuk CLIP, dll.
    def __init__(
        self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
        attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True,
        dims=2, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1,
        num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False,
        use_new_attention_order=False, pool="adaptive", l4q_params=None, *args, **kwargs ):
        super().__init__()
        self.in_channels = in_channels; self.model_channels = model_channels
        self.out_channels = out_channels; self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions; self.dropout = dropout
        self.channel_mult = channel_mult; self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint; self.dtype = th.float32 # LDM float32
        self.num_heads = num_heads; self.num_head_channels = num_head_channels
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            make_l4q_linear(model_channels, time_embed_dim, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs),
            nn.SiLU(),
            make_l4q_linear(time_embed_dim, time_embed_dim, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs),
        )
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(make_l4q_conv2d(dims, in_channels, model_channels, 3, padding=1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs))
        ])
        self._feature_size = model_channels; input_block_chans = [model_channels]; ch = model_channels; ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    curr_num_heads = num_heads if num_heads != -1 else (ch // num_head_channels if num_head_channels != -1 else 1)
                    if curr_num_heads == 0: curr_num_heads = 1 # Safety
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=curr_num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order, l4q_params=self.l4q_params))
                self.input_blocks.append(TimestepEmbedSequential(*layers)); self._feature_size += ch; input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch_down = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch_down, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True, l4q_params=self.l4q_params)
                    if resblock_updown else DownsampleOpenAI(ch, conv_resample, dims=dims, out_channels=out_ch_down, l4q_params=self.l4q_params)
                )); ch = out_ch_down; input_block_chans.append(ch); ds *= 2; self._feature_size += ch
        
        mid_num_heads = num_heads if num_heads != -1 else (ch // num_head_channels if num_head_channels != -1 else 1)
        if mid_num_heads == 0: mid_num_heads = 1

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=mid_num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order, l4q_params=self.l4q_params),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params),
        ); self._feature_size += ch
        
        self.pool = pool
        if pool == "adaptive":
            final_conv = make_l4q_conv2d(dims, ch, out_channels, 1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), nn.AdaptiveAvgPool2d((1, 1)), zero_module(final_conv), nn.Flatten())
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels, l4q_params=self.l4q_params))
        elif pool == "spatial":
            lin1 = make_l4q_linear(self._feature_size, 2048, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            lin2 = make_l4q_linear(2048, self.out_channels, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            self.out = nn.Sequential(lin1, nn.ReLU(), lin2)
        elif pool == "spatial_v2":
            lin1_v2 = make_l4q_linear(self._feature_size, 2048, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            lin2_v2 = make_l4q_linear(2048, self.out_channels, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            self.out = nn.Sequential(lin1_v2, normalization(2048), nn.SiLU(), lin2_v2)
        else: raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self): self.input_blocks.apply(convert_module_to_f16); self.middle_block.apply(convert_module_to_f16)
    def convert_to_fp32(self): self.input_blocks.apply(convert_module_to_f32); self.middle_block.apply(convert_module_to_f32)
    def forward(self, x, timesteps, context=None): # Tambah context untuk konsistensi API
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []; h = x.type(self.dtype)
        for module in self.input_blocks: h = module(h, emb, context); # Teruskan context
        if self.pool.startswith("spatial"): results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb, context) # Teruskan context
        if self.pool.startswith("spatial"): results.append(h.type(x.dtype).mean(dim=(2, 3))); h = th.cat(results, axis=-1); return self.out(h)
        else: h = h.type(x.dtype); return self.out(h)
