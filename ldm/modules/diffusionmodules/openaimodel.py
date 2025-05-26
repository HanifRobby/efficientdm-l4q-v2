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
    # conv_nd dan linear akan diganti dengan versi L4Q jika aktif
    # avg_pool_nd, zero_module, normalization, timestep_embedding tetap dipakai
    avg_pool_nd, 
    zero_module, # zero_module akan diterapkan ke layer L4Q juga
    normalization, 
    timestep_embedding,
)
# SpatialTransformer akan dimodifikasi untuk menerima l4q_params
from ldm.modules.attention import SpatialTransformer 


# --- Impor Fungsi Helper L4Q ---
# Idealnya, helper ini ada di l4q_utils.py yang terpusat
try:
    from ..l4q.l4q_utils import make_l4q_linear, make_l4q_conv2d
except ImportError:
    # Fallback jika impor langsung tidak berhasil. Ini perlu diperbaiki dengan path yang benar.
    # Untuk sementara, kita bisa mendefinisikan placeholder atau mencoba dari model.py jika helper ada di sana.
    try:
        from .model import make_l4q_linear, make_l4q_conv2d # Jika helper ada di model.py
        print("PERINGATAN di openaimodel.py: Mengimpor make_l4q_linear/conv2d dari .model, pastikan ini benar.")
    except ImportError:
        print("PERINGATAN KRITIKAL di openaimodel.py: Gagal mengimpor make_l4q_linear/conv2d. Layer L4Q TIDAK AKAN BERFUNGSI.")
        # Definisikan placeholder agar kode tidak langsung error, tapi ini HARUS DIPERBAIKI.
        def make_l4q_linear(in_features, out_features, bias=True, **kwargs):
            print("make_l4q_linear (placeholder): Menggunakan nn.Linear karena impor L4Q gagal.")
            return nn.Linear(in_features, out_features, bias=bias)
        def make_l4q_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
            print("make_l4q_conv2d (placeholder): Menggunakan nn.Conv2d karena impor L4Q gagal.")
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

# Fungsi conv_nd dan linear dari util.py (jika L4Q tidak aktif)
# Kita akan menggantinya secara dinamis
def original_conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def original_linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


# dummy replace (tidak berubah)
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(
        self,
        spacial_dim: int, # Nama variabel diubah dari spacial_dim ke spatial_dim untuk konsistensi
        embed_dim: int,
        num_heads_channels: int, # Ini adalah dim_head
        output_dim: int = None,
        l4q_params: dict = None, # Tambahkan l4q_params
    ):
        super().__init__()
        self.l4q_params = l4q_params
        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        self.positional_embedding = nn.Parameter(th.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5) # Dimensi dibalik
        
        # conv_nd(1, ...) berarti ini adalah Conv1d yang beroperasi pada sequence
        if l4q_enabled:
            # Asumsikan make_l4q_conv1d ada atau make_l4q_conv2d(..., kernel_size=(1,ks)) bisa dipakai jika kernel 1D
            # Untuk AttentionPool2d, qkv_proj dan c_proj adalah conv1d. Kita akan menggunakan linear jika itu lebih sesuai.
            # Dimensi input ke qkv_proj adalah embed_dim.
            # Jika ini benar-benar dimaksudkan sebagai conv1d yang beroperasi pada sequence (HW+1) dengan channel embed_dim:
            self.qkv_proj = make_l4q_conv2d(embed_dim, 3 * embed_dim, kernel_size=1, # Conv1D sebagai Conv2D dengan kernel (1,1) atau (1,K)
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.c_proj = make_l4q_conv2d(embed_dim, output_dim or embed_dim, kernel_size=1,
                                          lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.qkv_proj = original_conv_nd(1, embed_dim, 3 * embed_dim, 1) # Ini adalah nn.Conv1d
            self.c_proj = original_conv_nd(1, embed_dim, output_dim or embed_dim, 1) # Ini adalah nn.Conv1d
            
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads) # QKVAttention tidak punya parameter L4Q

    def forward(self, x):
        b, c, h, w = x.shape # x adalah (B, C, H, W)
        x = x.reshape(b, c, h * w) # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # Positional embedding: (HW+1, C)
        # x: (B, C, HW+1) -> perlu permute untuk penambahan
        x = x.permute(0, 2, 1) # B (HW+1) C
        x = x + self.positional_embedding.to(x.dtype)  # B (HW+1) C
        x = x.permute(0, 2, 1) # B C (HW+1)
        
        x = self.qkv_proj(x) # Input ke conv1d adalah (N, C_in, L_in)
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
            elif isinstance(layer, SpatialTransformer): # SpatialTransformer akan dimodifikasi untuk context
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UpsampleOpenAI(nn.Module): # Ganti nama agar tidak konflik dengan Upsample di model.py
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.l4q_params = l4q_params
        
        if use_conv:
            l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
            if l4q_enabled:
                self.conv = make_l4q_conv2d(self.channels, self.out_channels, kernel_size=3, padding=padding,
                                            lora_rank=self.l4q_params.get("lora_rank",4),
                                            n_bits=self.l4q_params.get("n_bits",4),
                                            alpha=self.l4q_params.get("alpha",1.0),
                                            quant_group_size=self.l4q_params.get("quant_group_size",-1))
            else:
                self.conv = original_conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module): # Tidak ada parameter L4Q di sini, karena hanya ConvTranspose2d
    def __init__(self, channels, out_channels=None, ks=5): # Tidak dimodifikasi untuk L4Q karena ConvTranspose
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)

class DownsampleOpenAI(nn.Module): # Ganti nama
    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1, l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.l4q_params = l4q_params
        stride = 2 if dims != 3 else (1, 2, 2)
        
        if use_conv:
            l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
            if l4q_enabled:
                self.op = make_l4q_conv2d(self.channels, self.out_channels, kernel_size=3, stride=stride, padding=padding,
                                          lora_rank=self.l4q_params.get("lora_rank",4),
                                          n_bits=self.l4q_params.get("n_bits",4),
                                          alpha=self.l4q_params.get("alpha",1.0),
                                          quant_group_size=self.l4q_params.get("quant_group_size",-1))
            else:
                self.op = original_conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False, # Untuk skip connection
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        l4q_params=None, # Tambahkan l4q_params
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv_skip = use_conv # Ganti nama agar tidak bentrok dengan use_conv untuk up/down
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        if l4q_enabled:
            conv1 = make_l4q_conv2d(dims, channels, self.out_channels, 3, padding=1,
                                    lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            conv1 = original_conv_nd(dims, channels, self.out_channels, 3, padding=1)
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv1)

        self.updown = up or down
        if up: # Upsample/Downsample di sini tidak menggunakan L4Q secara default di OpenAI, hanya interpolasi
            self.h_upd = UpsampleOpenAI(channels, False, dims, l4q_params=None) # False berarti tidak ada conv
            self.x_upd = UpsampleOpenAI(channels, False, dims, l4q_params=None)
        elif down:
            self.h_upd = DownsampleOpenAI(channels, False, dims, l4q_params=None)
            self.x_upd = DownsampleOpenAI(channels, False, dims, l4q_params=None)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if l4q_enabled:
            emb_linear = make_l4q_linear(emb_channels, 
                                         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                                         lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        else:
            emb_linear = original_linear(emb_channels,
                                 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        self.emb_layers = nn.Sequential(nn.SiLU(), emb_linear)
        
        if l4q_enabled:
            out_conv = make_l4q_conv2d(dims, self.out_channels, self.out_channels, 3, padding=1,
                                       lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            out_conv = original_conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(out_conv), # zero_module diterapkan ke layer L4Q juga
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif self.use_conv_skip: # Menggunakan self.use_conv_skip yang sudah diganti namanya
            if l4q_enabled:
                self.skip_connection = make_l4q_conv2d(dims, channels, self.out_channels, 3, padding=1,
                                                       lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            else:
                self.skip_connection = original_conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            if l4q_enabled:
                self.skip_connection = make_l4q_conv2d(dims, channels, self.out_channels, 1,
                                                       lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            else:
                self.skip_connection = original_conv_nd(dims, channels, self.out_channels, 1)

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
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
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
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False, # Tidak dipakai di LDM?
        l4q_params=None, # Tambahkan l4q_params
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, \
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        # QKV dan proj_out adalah conv1d yang beroperasi pada sequence (setelah reshape)
        if l4q_enabled:
            # Menggunakan conv2d dengan kernel 1x1 sebagai pengganti conv1d untuk L4Q
            self.qkv = make_l4q_conv2d(1, channels, channels * 3, kernel_size=1, # dims=1
                                       lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            proj_out_layer = make_l4q_conv2d(1, channels, channels, kernel_size=1, # dims=1
                                             lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.qkv = original_conv_nd(1, channels, channels * 3, 1)
            proj_out_layer = original_conv_nd(1, channels, channels, 1)
        
        if use_new_attention_order: # use_new_attention_order tidak ada di LDM UNetModel
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(proj_out_layer)


    def forward(self, x):
        # Checkpoint di OpenAI UNetModel biasanya di level ResBlock atau keseluruhan forward UNet.
        # Jika use_checkpoint di sini True, pastikan tidak ada konflik.
        # Kode asli LDM tidak memakai checkpoint di sini.
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_reshaped = x.reshape(b, c, -1) # Reshape ke (b, c, hw)
        qkv = self.qkv(self.norm(x_reshaped)) # qkv adalah conv1d, input (b,c,hw)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_reshaped + h).reshape(b, c, *spatial)


# --- QKVAttentionLegacy dan QKVAttention tidak memiliki parameter yang dapat dilatih L4Q ---
# Jadi, kelas-kelas ini tidak perlu dimodifikasi untuk l4q_params
class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
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

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
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

def count_flops_attn(model, _x, y): # Helper, tidak berubah
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    def __init__(
        self,
        image_size, # Tidak dipakai secara langsung di sini, tapi mungkin untuk attention pool
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True, # Untuk Upsample/Downsample OpenAI
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False, # Tidak dipakai di LDM
        num_heads=-1, # Default LDM
        num_head_channels=-1, # Default LDM
        num_heads_upsample=-1, # Tidak dipakai di LDM
        use_scale_shift_norm=False, # Default LDM
        resblock_updown=False, # Default LDM
        use_new_attention_order=False, # Tidak dipakai di LDM, AttentionBlock OpenAI berbeda
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None, # Untuk VQ
        legacy=True, # Tidak dipakai secara signifikan di LDM
        l4q_params=None, # <<< TAMBAHKAN INI >>>
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Context dim must be passed for spatial transformer'
        
        # Penyesuaian num_heads dan dim_head seperti di LDM model.py
        if num_heads == -1: # Jika num_heads tidak diset, hitung dari num_head_channels
            assert num_head_channels != -1
            # Ini akan dihitung per layer di AttentionBlock/SpatialTransformer
        if num_head_channels == -1: # Jika dim_head tidak diset, hitung dari num_heads
            assert num_heads != -1
            # Ini akan dihitung per layer

        self.image_size = image_size # Disimpan untuk referensi
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample # Untuk Upsample/Downsample OpenAI
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32 # LDM biasanya float32
        self.num_heads = num_heads # Disimpan untuk SpatialTransformer
        self.num_head_channels = num_head_channels # Disimpan untuk SpatialTransformer
        # self.num_heads_upsample = num_heads_upsample # Tidak dipakai di LDM
        self.predict_codebook_ids = n_embed is not None
        self.l4q_params = l4q_params # Simpan l4q_params

        # Ekstrak parameter L4Q
        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        time_embed_dim = model_channels * 4
        
        # Time embedding layers
        if l4q_enabled:
            time_emb_linear1 = make_l4q_linear(model_channels, time_embed_dim, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            time_emb_linear2 = make_l4q_linear(time_embed_dim, time_embed_dim, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        else:
            time_emb_linear1 = original_linear(model_channels, time_embed_dim)
            time_emb_linear2 = original_linear(time_embed_dim, time_embed_dim)
        self.time_embed = nn.Sequential(time_emb_linear1, nn.SiLU(), time_emb_linear2)

        if self.num_classes is not None:
            if l4q_enabled: # Asumsikan label_emb juga bisa L4Q jika diinginkan, tapi biasanya tidak
                self.label_emb = nn.Embedding(num_classes, time_embed_dim) # Embedding biasanya tidak dikuantisasi dengan LoRA
            else:
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Input blocks
        if l4q_enabled:
            conv_in_layer = make_l4q_conv2d(dims, in_channels, model_channels, 3, padding=1,
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            conv_in_layer = original_conv_nd(dims, in_channels, model_channels, 3, padding=1)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_in_layer)])
        
        self._feature_size = model_channels # Tidak dipakai di LDM?
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=mult * model_channels,
                        dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                        l4q_params=self.l4q_params # Teruskan l4q_params
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # Menentukan dim_head untuk SpatialTransformer atau AttentionBlock
                    if num_head_channels == -1 and num_heads != -1 :
                        current_dim_head = ch // num_heads
                    elif num_head_channels != -1:
                        current_dim_head = num_head_channels
                    else: # Fallback jika keduanya -1, atau salah satu tidak valid
                        current_dim_head = 64 # Default LDM
                    
                    # Menentukan num_heads untuk SpatialTransformer atau AttentionBlock
                    if num_heads != -1:
                        current_num_heads = num_heads
                    elif num_head_channels != -1:
                        current_num_heads = ch // num_head_channels
                    else:
                        current_num_heads = 8 # Default LDM

                    layers.append(
                        SpatialTransformer(
                            ch, current_num_heads, current_dim_head, depth=transformer_depth, context_dim=context_dim,
                            l4q_params=self.l4q_params # Teruskan l4q_params
                        ) if use_spatial_transformer else AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=current_num_heads, # num_heads dari argumen UNet
                            num_head_channels=num_head_channels, # num_head_channels dari argumen UNet
                            l4q_params=self.l4q_params # Teruskan l4q_params
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch # Tidak dipakai di LDM
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch_down = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock( # ResBlock untuk downsampling jika resblock_updown=True
                            ch, time_embed_dim, dropout, out_channels=out_ch_down, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True,
                            l4q_params=self.l4q_params
                        ) if resblock_updown else DownsampleOpenAI( # Downsample OpenAI
                            ch, conv_resample, dims=dims, out_channels=out_ch_down, l4q_params=self.l4q_params
                        )
                    )
                )
                ch = out_ch_down
                input_block_chans.append(ch)
                ds *= 2
                # self._feature_size += ch # Tidak dipakai

        # Middle block
        # Menentukan dim_head dan num_heads untuk middle block
        if num_head_channels == -1 and num_heads != -1: mid_dim_head = ch // num_heads
        elif num_head_channels != -1: mid_dim_head = num_head_channels
        else: mid_dim_head = 64 
        
        if num_heads != -1: mid_num_heads = num_heads
        elif num_head_channels != -1: mid_num_heads = ch // num_head_channels
        else: mid_num_heads = 8

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params
            ),
            SpatialTransformer(
                ch, mid_num_heads, mid_dim_head, depth=transformer_depth, context_dim=context_dim,
                l4q_params=self.l4q_params
            ) if use_spatial_transformer else AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=mid_num_heads, 
                num_head_channels=num_head_channels, # Gunakan num_head_channels dari argumen UNet
                l4q_params=self.l4q_params
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params
            ),
        )
        # self._feature_size += ch # Tidak dipakai

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult,
                        dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                        l4q_params=self.l4q_params
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1 and num_heads != -1: current_dim_head = ch // num_heads
                    elif num_head_channels != -1: current_dim_head = num_head_channels
                    else: current_dim_head = 64
                    
                    if num_heads != -1: current_num_heads = num_heads
                    elif num_head_channels != -1: current_num_heads = ch // num_head_channels
                    else: current_num_heads = 8
                        
                    layers.append(
                        SpatialTransformer(
                            ch, current_num_heads, current_dim_head, depth=transformer_depth, context_dim=context_dim,
                            l4q_params=self.l4q_params
                        ) if use_spatial_transformer else AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=current_num_heads, # num_heads_upsample tidak ada di LDM
                            num_head_channels=num_head_channels, # Gunakan num_head_channels dari argumen UNet
                            l4q_params=self.l4q_params
                        )
                    )
                if level and i == num_res_blocks: # Hanya upsample jika bukan level terendah (level > 0)
                    out_ch_up = ch
                    layers.append(
                        ResBlock( # ResBlock untuk upsampling jika resblock_updown=True
                            ch, time_embed_dim, dropout, out_channels=out_ch_up, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True,
                            l4q_params=self.l4q_params
                        ) if resblock_updown else UpsampleOpenAI( # Upsample OpenAI
                            ch, conv_resample, dims=dims, out_channels=out_ch_up, l4q_params=self.l4q_params
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # self._feature_size += ch # Tidak dipakai

        # Output convolution
        if l4q_enabled:
            out_conv_layer = make_l4q_conv2d(dims, ch, out_channels, 3, padding=1, # ch adalah model_channels di sini
                                             lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            out_conv_layer = original_conv_nd(dims, ch, out_channels, 3, padding=1)
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(out_conv_layer), # zero_module diterapkan ke layer L4Q juga
        )
        
        if self.predict_codebook_ids: # Untuk VQModel
            if l4q_enabled:
                id_pred_conv = make_l4q_conv2d(dims, model_channels, n_embed, 1,
                                               lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            else:
                id_pred_conv = original_conv_nd(dims, model_channels, n_embed, 1)
            self.id_predictor = nn.Sequential(normalization(ch), id_pred_conv)


    def convert_to_fp16(self): # Tidak dipakai di LDM
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self): # Tidak dipakai di LDM
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"
        hs = []
        # Timestep embedding
        if timesteps is None: # Jika tidak ada timestep (misalnya untuk VAE encoder/decoder)
            # Buat tensor dummy nol jika diperlukan oleh ResBlock, atau pastikan ResBlock bisa handle temb=None
            # Di LDM, UNetModel selalu dipanggil dengan timesteps. Encoder/Decoder VAE tidak pakai UNetModel ini.
            # Jika ini UNet untuk difusi, timesteps harus ada.
            raise ValueError("Timesteps must be provided to UNetModel forward pass.")

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype) # x adalah input latent
        # Input blocks
        for module in self.input_blocks:
            h = module(h, emb, context) # TimestepEmbedSequential akan meneruskan context ke SpatialTransformer
            hs.append(h)
        # Middle block
        h = self.middle_block(h, emb, context)
        # Output blocks
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class EncoderUNetModel(nn.Module): # UNet Encoder untuk CLIP image embeddings, dll.
    def __init__(
        self,
        image_size, # Untuk AttentionPool2d
        in_channels,
        model_channels,
        out_channels, # Output feature dimension
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        # use_fp16=False, # Tidak dipakai
        num_heads=1, # Berbeda dari UNetModel utama
        num_head_channels=-1,
        # num_heads_upsample=-1, # Tidak ada upsampling
        use_scale_shift_norm=False,
        resblock_updown=False, # Hanya downsampling
        # use_new_attention_order=False, # Tidak dipakai
        pool="adaptive", # Jenis pooling di akhir
        l4q_params=None, # <<< TAMBAHKAN INI >>>
        *args, **kwargs # Tangkap argumen ekstra
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32 # LDM biasanya float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        time_embed_dim = model_channels * 4
        if l4q_enabled:
            time_emb_linear1 = make_l4q_linear(model_channels, time_embed_dim, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            time_emb_linear2 = make_l4q_linear(time_embed_dim, time_embed_dim, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        else:
            time_emb_linear1 = original_linear(model_channels, time_embed_dim)
            time_emb_linear2 = original_linear(time_embed_dim, time_embed_dim)
        self.time_embed = nn.Sequential(time_emb_linear1, nn.SiLU(), time_emb_linear2)
        
        if l4q_enabled:
            conv_in_layer = make_l4q_conv2d(dims, in_channels, model_channels, 3, padding=1,
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            conv_in_layer = original_conv_nd(dims, in_channels, model_channels, 3, padding=1)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_in_layer)])
        
        self._feature_size = model_channels # Untuk spatial pooling
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=mult * model_channels,
                        dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                        l4q_params=self.l4q_params
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # Tentukan num_heads dan dim_head untuk AttentionBlock
                    if num_head_channels == -1 and self.num_heads != -1 : current_num_heads = self.num_heads
                    elif num_head_channels != -1: current_num_heads = ch // num_head_channels
                    else: current_num_heads = 1 # Default OpenAI AttentionBlock

                    layers.append(
                        AttentionBlock( # OpenAI AttentionBlock, bukan SpatialTransformer
                            ch, use_checkpoint=use_checkpoint, num_heads=current_num_heads,
                            num_head_channels=num_head_channels, # num_head_channels dari argumen
                            l4q_params=self.l4q_params
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch_down = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch_down, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True,
                            l4q_params=self.l4q_params
                        ) if resblock_updown else DownsampleOpenAI(
                            ch, conv_resample, dims=dims, out_channels=out_ch_down, l4q_params=self.l4q_params
                        )
                    )
                )
                ch = out_ch_down
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Tentukan num_heads dan dim_head untuk middle block AttentionBlock
        if num_head_channels == -1 and self.num_heads != -1: mid_num_heads = self.num_heads
        elif num_head_channels != -1: mid_num_heads = ch // num_head_channels
        else: mid_num_heads = 1

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params
            ),
            AttentionBlock( # OpenAI AttentionBlock
                ch, use_checkpoint=use_checkpoint, num_heads=mid_num_heads,
                num_head_channels=num_head_channels, l4q_params=self.l4q_params
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm, l4q_params=self.l4q_params
            ),
        )
        self._feature_size += ch
        self.pool = pool
        
        if pool == "adaptive":
            final_conv_out_channels = out_channels
            if l4q_enabled:
                final_conv = make_l4q_conv2d(dims, ch, final_conv_out_channels, 1,
                                             lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            else:
                final_conv = original_conv_nd(dims, ch, final_conv_out_channels, 1)

            self.out = nn.Sequential(
                normalization(ch), nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(final_conv),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch), nn.SiLU(),
                AttentionPool2d( # AttentionPool2d juga perlu l4q_params
                    (image_size // ds), ch, num_head_channels, out_channels, l4q_params=self.l4q_params
                ),
            )
        elif pool == "spatial": # Menggunakan Linear
            if l4q_enabled:
                lin1 = make_l4q_linear(self._feature_size, 2048, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
                lin2 = make_l4q_linear(2048, self.out_channels, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            else:
                lin1 = original_linear(self._feature_size, 2048)
                lin2 = original_linear(2048, self.out_channels)
            self.out = nn.Sequential(lin1, nn.ReLU(), lin2)
        elif pool == "spatial_v2": # Menggunakan Linear
            if l4q_enabled:
                lin1_v2 = make_l4q_linear(self._feature_size, 2048, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
                lin2_v2 = make_l4q_linear(2048, self.out_channels, lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            else:
                lin1_v2 = original_linear(self._feature_size, 2048)
                lin2_v2 = original_linear(2048, self.out_channels)
            self.out = nn.Sequential(lin1_v2, normalization(2048), nn.SiLU(), lin2_v2)
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self): # Tidak dipakai
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self): # Tidak dipakai
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps, context=None): # Tambah context untuk konsistensi, meskipun tidak dipakai di sini
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context) # Teruskan context ke TimestepEmbedSequential
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb, context) # Teruskan context
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)