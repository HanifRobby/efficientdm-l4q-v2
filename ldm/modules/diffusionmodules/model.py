# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
# Pastikan LinearAttention tidak menggunakan nn.Linear/Conv2d yang ingin diganti
# atau sudah di-patch/dimodifikasi jika perlu.
from ldm.modules.attention import LinearAttention 

# Impor layer L4Q dan fungsi helper Anda
from ldm.modules.l4q.l4q_linear_layer import L4QQuantizedLinear
from ldm.modules.l4q.l4q_conv2d_layer import L4QQuantizedConv2d

# --- Fungsi Helper L4Q ---
def make_l4q_linear(in_features, out_features, bias=True, lora_rank=4, n_bits=4, alpha=1.0, group_size=-1):
    """Helper function to create L4QQuantizedLinear layer."""
    return L4QQuantizedLinear(
        in_features, out_features, bias=bias,
        lora_rank=lora_rank, n_bits=n_bits, alpha=alpha, 
        group_size=group_size
    )

def make_l4q_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                    lora_rank=4, n_bits=4, alpha=1.0, quant_group_size=-1):
    """Helper function to create L4QQuantizedConv2d layer."""
    return L4QQuantizedConv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        lora_rank=lora_rank, n_bits=n_bits, alpha=alpha, 
        quant_group_size=quant_group_size
    )

# --- Fungsi Utilitas Model ---
def get_timestep_embedding(timesteps, embedding_dim):
    """Build sinusoidal embeddings."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# --- Sub-Modul dengan Integrasi L4Q ---
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, l4q_params=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
            if l4q_enabled:
                self.conv = make_l4q_conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                            lora_rank=l4q_params.get("lora_rank", 4),
                                            n_bits=l4q_params.get("n_bits", 4),
                                            alpha=l4q_params.get("alpha", 1.0),
                                            quant_group_size=l4q_params.get("quant_group_size", -1))
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, l4q_params=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
            if l4q_enabled:
                self.conv = make_l4q_conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0, # Padding manual di forward
                                            lora_rank=l4q_params.get("lora_rank", 4),
                                            n_bits=l4q_params.get("n_bits", 4),
                                            alpha=l4q_params.get("alpha", 1.0),
                                            quant_group_size=l4q_params.get("quant_group_size", -1))
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, l4q_params=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.norm1 = Normalize(in_channels)
        if l4q_enabled:
            self.conv1 = make_l4q_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                         lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0: # temb_channels bisa 0 untuk Encoder/Decoder VAE
            self.temb_proj = nn.Linear(temb_channels, out_channels) # Biarkan Linear standar atau buat L4Q jika perlu
            if l4q_enabled and hasattr(self, 'temb_proj'): # Jika temb_proj ada dan L4Q aktif
                 self.temb_proj = make_l4q_linear(temb_channels, out_channels,
                                                 lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        if l4q_enabled:
            self.conv2 = make_l4q_conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                         lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if l4q_enabled:
                    self.conv_shortcut = make_l4q_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                         lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
                else:
                    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                if l4q_enabled:
                    self.nin_shortcut = make_l4q_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                                        lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
                else:
                    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and hasattr(self, 'temb_proj'):
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels, l4q_params=None): # Tambahkan l4q_params, meskipun LinearAttention mungkin tidak langsung menggunakannya
        # Jika LinearAttention secara internal menggunakan nn.Linear/Conv2d yang perlu di-patch,
        # maka LinearAttention itu sendiri perlu dimodifikasi.
        # Untuk sekarang, kita asumsikan LinearAttention tidak terpengaruh L4Q secara langsung.
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, l4q_params=None):
        super().__init__()
        self.in_channels = in_channels

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.norm = Normalize(in_channels)
        if l4q_enabled:
            self.q = make_l4q_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.k = make_l4q_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.v = make_l4q_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.proj_out = make_l4q_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k) * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b,c,h*w).permute(0,2,1) # Perlu permute v juga untuk bmm yang benar
        h_ = torch.bmm(w_,v) # Urutan bmm: (B, HW, HW) @ (B, HW, C) -> (B, HW, C)
        h_ = h_.permute(0,2,1).reshape(b,c,h,w) # Kembalikan ke shape gambar
        h_ = self.proj_out(h_)
        return x+h_

def make_attn(in_channels, attn_type="vanilla", l4q_params=None):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels, l4q_params=l4q_params)
    elif attn_type == "none":
        return nn.Identity()
    else: # linear attention
        return LinAttnBlock(in_channels, l4q_params=l4q_params) # Teruskan l4q_params ke LinAttnBlock

class Model(nn.Module): # Ini adalah UNetModel Anda
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla",
                 l4q_params=None):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.l4q_params = l4q_params 

        self.l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        self.lora_rank = l4q_params.get("lora_rank", 4) if l4q_params else 4
        self.n_bits = l4q_params.get("n_bits", 4) if l4q_params else 4
        self.alpha = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        self.quant_group_size = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            if self.l4q_enabled:
                self.temb.dense = nn.ModuleList([
                    make_l4q_linear(self.ch, self.temb_ch,
                                    lora_rank=self.lora_rank, n_bits=self.n_bits, 
                                    alpha=self.alpha, group_size=self.quant_group_size),
                    make_l4q_linear(self.temb_ch, self.temb_ch,
                                    lora_rank=self.lora_rank, n_bits=self.n_bits, 
                                    alpha=self.alpha, group_size=self.quant_group_size),
                ])
            else:
                self.temb.dense = nn.ModuleList([
                    nn.Linear(self.ch, self.temb_ch),
                    nn.Linear(self.temb_ch, self.temb_ch),
                ])

        if self.l4q_enabled:
            self.conv_in = make_l4q_conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1,
                                           lora_rank=self.lora_rank, n_bits=self.n_bits,
                                           alpha=self.alpha, quant_group_size=self.quant_group_size)
        else:
            self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout,
                                         l4q_params=self.l4q_params))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params))
            down_module = nn.Module()
            down_module.block = block
            down_module.attn = attn
            if i_level != self.num_resolutions-1:
                down_module.downsample = Downsample(block_in, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res // 2
            self.down.append(down_module)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # block_in untuk ResNetBlock di upsampling adalah kombinasi dari output level sebelumnya 
            # dan skip connection. Nilai awalnya adalah output dari middle block (block_in terakhir dari downsampling).
            current_block_in_upsampling = block_in # Ini adalah block_in dari akhir down/middle path

            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in_channels_actual = ch*in_ch_mult[i_level] 
                else:
                    skip_in_channels_actual = ch*ch_mult[i_level]
                
                input_to_resnet = current_block_in_upsampling + skip_in_channels_actual
                block.append(ResnetBlock(in_channels=input_to_resnet, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout,
                                         l4q_params=self.l4q_params))
                current_block_in_upsampling = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(current_block_in_upsampling, attn_type=attn_type, l4q_params=self.l4q_params))
            
            up_module = nn.Module()
            up_module.block = block
            up_module.attn = attn
            if i_level != 0:
                up_module.upsample = Upsample(current_block_in_upsampling, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res * 2
            self.up.insert(0, up_module)
            block_in = current_block_in_upsampling


        self.norm_out = Normalize(block_out)
        if self.l4q_enabled:
            self.conv_out = make_l4q_conv2d(block_out, out_ch, kernel_size=3, stride=1, padding=1,
                                            lora_rank=self.lora_rank, n_bits=self.n_bits,
                                            alpha=self.alpha, quant_group_size=self.quant_group_size)
        else:
            self.conv_out = nn.Conv2d(block_out, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None):
        if self.use_timestep:
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0: # AttnBlock dipanggil jika ada
                    # Asumsikan attn list memiliki panjang yang sama atau lebih dari block list
                    # atau logika indexing yang sesuai
                    if i_block < len(self.down[i_level].attn):
                         h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                skip_h = hs.pop()
                h_cat = torch.cat([h, skip_h], dim=1)
                h = self.up[i_level].block[i_block](h_cat, temb)
                if len(self.up[i_level].attn) > 0: # AttnBlock dipanggil jika ada
                    if i_block < len(self.up[i_level].attn):
                        h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        if hasattr(self.conv_out, 'w0'): 
            return self.conv_out.w0 
        else: 
            return self.conv_out.weight

# --- Kelas Encoder, Decoder, dll. dengan Integrasi L4Q ---
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 l4q_params=None, **ignore_kwargs): # Tambahkan l4q_params
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0 # Encoder biasanya tidak menggunakan timestep embedding
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.l4q_params = l4q_params # Simpan l4q_params

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        if l4q_enabled:
            self.conv_in = make_l4q_conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1,
                                           lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult # Tidak terpakai?
        self.down = nn.ModuleList()
        block_in_channels_for_loop = self.ch # Inisialisasi block_in untuk loop pertama
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # block_in dihitung ulang di setiap level berdasarkan ch dan in_ch_mult
            block_in = self.ch*in_ch_mult[i_level]
            block_out = self.ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout,
                                         l4q_params=self.l4q_params)) # Teruskan l4q_params
                block_in = block_out # Update block_in untuk iterasi berikutnya
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params))
            down_module = nn.Module()
            down_module.block = block
            down_module.attn = attn
            if i_level != self.num_resolutions-1:
                down_module.downsample = Downsample(block_in, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res // 2
            self.down.append(down_module)
            block_in_channels_for_loop = block_in # Simpan block_in terakhir untuk level berikutnya atau mid block

        self.mid = nn.Module()
        # block_in untuk mid adalah block_out dari level terakhir downsampling
        mid_block_channels = block_in_channels_for_loop
        self.mid.block_1 = ResnetBlock(in_channels=mid_block_channels, out_channels=mid_block_channels,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(mid_block_channels, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=mid_block_channels, out_channels=mid_block_channels,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        self.norm_out = Normalize(mid_block_channels)
        if l4q_enabled:
            self.conv_out = make_l4q_conv2d(mid_block_channels, 2*z_channels if double_z else z_channels,
                                            kernel_size=3, stride=1, padding=1,
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv_out = nn.Conv2d(mid_block_channels, 2*z_channels if double_z else z_channels,
                                      kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None # Encoder tidak menggunakan timestep embedding
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                     if i_block < len(self.down[i_level].attn): # Defensive check
                        h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, # in_channels tidak dipakai di sini
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", l4q_params=None, **ignorekwargs): # Tambahkan l4q_params
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0 # Decoder VAE biasanya tidak pakai temb
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        # self.in_channels = in_channels # Tidak dipakai, input adalah z_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.l4q_params = l4q_params

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        in_ch_mult = (1,)+tuple(ch_mult) # Tidak dipakai di Decoder?
        block_in = ch*ch_mult[self.num_resolutions-1] # Channel di resolusi terendah (setelah conv_in)
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res) # Untuk print info
        # print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        if l4q_enabled:
            self.conv_in = make_l4q_conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1,
                                           lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        self.up = nn.ModuleList()
        # block_in di sini adalah output dari mid block
        current_block_in_upsampling = block_in
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=current_block_in_upsampling, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params))
                current_block_in_upsampling = block_out # Update untuk iterasi block berikutnya
                if curr_res in attn_resolutions:
                    attn.append(make_attn(current_block_in_upsampling, attn_type=attn_type, l4q_params=self.l4q_params))
            up_module = nn.Module()
            up_module.block = block
            up_module.attn = attn
            if i_level != 0:
                up_module.upsample = Upsample(current_block_in_upsampling, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res * 2
            self.up.insert(0, up_module)
            # current_block_in_upsampling sudah menjadi block_out dari ResNetBlock terakhir di level ini,
            # atau output dari upsample jika ada.

        self.norm_out = Normalize(current_block_in_upsampling) # block_in terakhir adalah output dari loop
        if l4q_enabled:
            self.conv_out = make_l4q_conv2d(current_block_in_upsampling, out_ch, kernel_size=3, stride=1, padding=1,
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv_out = nn.Conv2d(current_block_in_upsampling, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    if i_block < len(self.up[i_level].attn): # Defensive
                        h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, l4q_params=None, *args, **kwargs): # Tambahkan l4q_params
        super().__init__()
        self.l4q_params = l4q_params
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        if l4q_enabled:
            conv1 = make_l4q_conv2d(in_channels, in_channels, 1, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            conv2 = make_l4q_conv2d(2*in_channels, in_channels, 1, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.conv_out_l4q = make_l4q_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            conv1 = nn.Conv2d(in_channels, in_channels, 1)
            conv2 = nn.Conv2d(2*in_channels, in_channels, 1)
            self.conv_out_std = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


        self.model = nn.ModuleList([conv1,
                                     ResnetBlock(in_channels=in_channels, out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0, l4q_params=self.l4q_params),
                                     ResnetBlock(in_channels=2 * in_channels, out_channels=4 * in_channels,
                                                 temb_channels=0, dropout=0.0, l4q_params=self.l4q_params),
                                     ResnetBlock(in_channels=4 * in_channels, out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0, l4q_params=self.l4q_params),
                                     conv2,
                                     Upsample(in_channels, with_conv=True, l4q_params=self.l4q_params)])
        self.norm_out = Normalize(in_channels)
        # conv_out sudah diinisialisasi di atas berdasarkan l4q_enabled

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]: # ResnetBlocks
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        if hasattr(self, 'conv_out_l4q'):
            x = self.conv_out_l4q(h)
        else:
            x = self.conv_out_std(h)
        return x

class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0, l4q_params=None): # Tambahkan l4q_params
        super().__init__()
        self.l4q_params = l4q_params
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        # curr_res = resolution // 2 ** (self.num_resolutions - 1) # Tidak dipakai
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block_level = nn.ModuleList() # Ganti nama variabel agar tidak konflik
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1): # i_block tidak dipakai
                res_block_level.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                                 temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params))
                block_in = block_out
            self.res_blocks.append(res_block_level)
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True, l4q_params=self.l4q_params))
                # curr_res = curr_res * 2 # Tidak dipakai

        self.norm_out = Normalize(block_in)
        if l4q_enabled:
            self.conv_out = make_l4q_conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1,
                                            lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        for k, i_level_idx in enumerate(range(self.num_resolutions)): # Ganti nama variabel
            for i_block_idx in range(self.num_res_blocks + 1): # Ganti nama variabel
                h = self.res_blocks[i_level_idx][i_block_idx](h, None)
            if i_level_idx != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h) # k adalah index untuk upsample_blocks
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2, l4q_params=None): # Tambahkan l4q_params
        super().__init__()
        self.factor = factor
        self.l4q_params = l4q_params
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        if l4q_enabled:
            self.conv_in = make_l4q_conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                                           lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            self.conv_out_l4q = make_l4q_conv2d(mid_channels, out_channels, kernel_size=1,
                                                lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
            self.conv_out_std = nn.Conv2d(mid_channels, out_channels, kernel_size=1)


        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                                     temb_channels=0, dropout=0.0, l4q_params=self.l4q_params) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels, l4q_params=self.l4q_params) # AttnBlock sudah diubah
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                                     temb_channels=0, dropout=0.0, l4q_params=self.l4q_params) for _ in range(depth)])
        # conv_out sudah diinisialisasi di atas

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x) # AttnBlock dipanggil di sini
        for block in self.res_block2:
            x = block(x, None)
        
        if hasattr(self, 'conv_out_l4q'):
            x = self.conv_out_l4q(x)
        else:
            x = self.conv_out_std(x)
        return x

class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1, l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.l4q_params = l4q_params # Simpan untuk diteruskan
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None, l4q_params=self.l4q_params) # Teruskan l4q_params
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth,
                                       l4q_params=self.l4q_params) # Teruskan l4q_params

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x

class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1, l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.l4q_params = l4q_params # Simpan untuk diteruskan
        tmp_chn = z_channels # Di kode asli, tmp_chn = z_channels*ch_mult[-1], ini mungkin typo jika z_channels sudah output dari rescaler
                            # Mari asumsikan z_channels adalah input ke rescaler, dan output rescaler adalah tmp_chn
        
        # Jika z_channels adalah input ke rescaler, dan output rescaler adalah input ke decoder:
        rescaler_out_channels = z_channels * ch_mult[-1] # Asumsi ini adalah channel setelah rescaler
        
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=rescaler_out_channels, # mid_channels bisa sama dengan out jika tidak ada perubahan channel di rescaler
                                       out_channels=rescaler_out_channels, depth=rescale_module_depth,
                                       l4q_params=self.l4q_params) # Teruskan l4q_params
        self.decoder = Decoder(out_ch=out_ch, z_channels=rescaler_out_channels, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch, l4q_params=self.l4q_params) # Teruskan l4q_params

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x

class Upsampler(nn.Module): # Digunakan oleh image_editing.PaintByExample
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2, l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.l4q_params = l4q_params
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1 if out_size > in_size else 1
        factor_up = float(out_size) / in_size # Harus float untuk interpolasi
        
        # print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        # LatentRescaler di sini mungkin tidak diperlukan jika factor_up = 1.0
        # Atau jika tujuannya hanya upsampling resolusi tanpa mengubah channel secara signifikan sebelum decoder.
        # Asumsikan mid_channels di rescaler bisa sama dengan in_channels jika hanya scaling.
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, 
                                       mid_channels=in_channels * ch_mult if factor_up > 1.0 else in_channels, # Hanya tambah channel jika ada upscaling
                                       out_channels=in_channels, depth=1, # depth=1 untuk rescaler sederhana
                                       l4q_params=self.l4q_params)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels, # ch untuk decoder bisa in_channels
                               ch_mult=[ch_mult for _ in range(num_blocks)], l4q_params=self.l4q_params)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x

class Resize(nn.Module): # Tidak banyak digunakan, dan implementasi conv-nya di-raise NotImplementedError
    def __init__(self, in_channels=None, learned=False, mode="bilinear", l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        self.l4q_params = l4q_params # Simpan jika diperlukan
        if self.with_conv:
            # print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError("Learned Resize with L4Q not fully implemented/verified here.")
            # assert in_channels is not None
            # l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
            # if l4q_enabled:
            #     self.conv = make_l4q_conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
            #                                 lora_rank=l4q_params.get("lora_rank",4), n_bits=l4q_params.get("n_bits",4),
            #                                 alpha=l4q_params.get("alpha",1.0), quant_group_size=l4q_params.get("quant_group_size",-1))
            # else:
            #     self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)


    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):
    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None, # Channel output dari proj, dan input ke ResNet pertama
                 dropout=0.,
                 pretrained_config=None,
                 l4q_params=None): # Tambah l4q_params
        super().__init__()
        self.l4q_params = l4q_params
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            # Coba dapatkan dari encoder VAE jika ada, jika tidak, default ke in_channels
            try:
                n_channels = self.pretrained_model.encoder.ch 
            except AttributeError:
                n_channels = in_channels


        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2 if in_channels >=2 else 1) # num_groups harus bisa membagi in_channels

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        if l4q_enabled:
            self.proj = make_l4q_conv2d(in_channels,n_channels,kernel_size=3, stride=1,padding=1,
                                        lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        else:
            self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3, stride=1,padding=1)


        blocks = []
        downs = []
        ch_in = n_channels
        for m_idx, m in enumerate(ch_mult): # Ganti nama variabel agar tidak konflik
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout, temb_channels=0, l4q_params=self.l4q_params))
            ch_in = m * n_channels
            # Downsample hanya jika bukan iterasi terakhir dari ch_mult
            if m_idx < len(ch_mult) -1 :
                downs.append(Downsample(ch_in, with_conv=False, l4q_params=self.l4q_params)) # with_conv=False tidak pakai conv
            else: # Untuk iterasi terakhir, tidak ada downsample
                downs.append(nn.Identity())


        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_with_pretrained(self,x):
        # Periksa apakah VAE memiliki DiagonalGaussianDistribution
        if hasattr(self.pretrained_model, 'encode') and callable(self.pretrained_model.encode):
            c = self.pretrained_model.encode(x)
            if hasattr(c, 'mode') and callable(c.mode): # Cek jika outputnya adalah DiagonalGaussianDistribution
                c = c.mode()
            return c
        else: # Jika VAE tidak memiliki metode encode standar (misalnya, hanya model VQ)
            raise NotImplementedError("Pretrained model does not have a standard 'encode' method or 'DiagonalGaussianDistribution' output.")


    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for i in range(len(self.model)):
            z = self.model[i](z,temb=None) # ResnetBlock
            z = self.downsampler[i](z)     # Downsample atau nn.Identity

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z
