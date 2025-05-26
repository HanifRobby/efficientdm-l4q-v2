# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention

from ldm.modules.l4q.l4q_linear_layer import L4QQuantizedLinear
from ldm.modules.l4q.l4q_conv2d_layer import L4QQuantizedConv2d

def make_l4q_linear(in_features, out_features, bias=True, lora_rank=4, n_bits=4, alpha=1.0, group_size=-1):
    # Di sini Anda bisa menambahkan logika default atau konfigurasi khusus
    # untuk semua layer L4Q yang dibuat di model ini.
    # Contoh: mengambil lora_rank, n_bits dari konfigurasi global
    # atau meneruskannya sebagai argumen jika berbeda per layer.
    return L4QQuantizedLinear(
        in_features,
        out_features,
        bias=bias,
        lora_rank=lora_rank, # Ambil dari config atau argumen
        n_bits=n_bits,       # Ambil dari config atau argumen
        alpha=alpha,         # Ambil dari config atau argumen
        group_size=group_size  # Ambil dari config atau argumen
    )

def make_l4q_conv2d(in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True,
                    # Parameter L4Q spesifik:
                    lora_rank=4, # Default atau ambil dari config
                    n_bits=4,    # Default atau ambil dari config
                    alpha=1.0,   # Default atau ambil dari config
                    quant_group_size=-1 # Default atau ambil dari config
                    ):
    return L4QQuantizedConv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        lora_rank=lora_rank,
        n_bits=n_bits,
        alpha=alpha,
        quant_group_size=quant_group_size
    )

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
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


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, l4q_params=None): # Tambahkan l4q_params
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            if l4q_params and l4q_params.get("enabled", False):
                self.conv = make_l4q_conv2d(in_channels,
                                            in_channels,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
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
    def __init__(self, in_channels, with_conv, l4q_params=None): # Tambahkan l4q_params
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            if l4q_params and l4q_params.get("enabled", False):
                self.conv = make_l4q_conv2d(in_channels,
                                            in_channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0, # Padding manual di forward
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
                 dropout, temb_channels=512, l4q_params=None): # Tambahkan l4q_params
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

        if temb_channels > 0:
            if l4q_enabled:
                self.temb_proj = make_l4q_linear(temb_channels, out_channels,
                                                 lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            else:
                self.temb_proj = nn.Linear(temb_channels, out_channels)
        
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

        if temb is not None and hasattr(self, 'temb_proj'): # Pastikan temb_proj ada
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
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, l4q_params=None): # Tambahkan l4q_params
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
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)
        h_ = self.proj_out(h_)
        return x+h_


def make_attn(in_channels, attn_type="vanilla", l4q_params=None): # Tambahkan l4q_params
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, l4q_params=l4q_params) # Teruskan l4q_params
    elif attn_type == "none":
        return nn.Identity() # nn.Identity tidak memiliki parameter in_channels
    else: # linear attention
        # Pastikan LinAttnBlock juga di-patch atau dimodifikasi jika mengandung nn.Linear/Conv2d
        # Untuk sekarang, asumsikan LinearAttention dari ldm.modules.attention tidak perlu L4Q
        # atau sudah di-patch secara terpisah jika perlu. Jika ia memanggil nn.Linear/Conv2d secara internal,
        # maka itu perlu di-patch atau LinearAttention itu sendiri perlu parameter l4q_params.
        # Jika LinAttnBlock tidak menggunakan conv/linear yang perlu dikuantisasi, biarkan seperti ini.
        return LinAttnBlock(in_channels)


class Model(nn.Module): # Ini adalah UNetModel Anda
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla",
                 # >>> TAMBAHKAN l4q_params di sini <<<
                 l4q_params=None):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.l4q_params = l4q_params # Simpan untuk digunakan oleh sub-modul jika diperlukan

        # Ekstrak parameter L4Q dengan nilai default jika l4q_params tidak ada atau key tidak ada
        self.l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        self.lora_rank = l4q_params.get("lora_rank", 4) if l4q_params else 4
        self.n_bits = l4q_params.get("n_bits", 4) if l4q_params else 4
        self.alpha = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        self.quant_group_size = l4q_params.get("quant_group_size", -1) if l4q_params else -1
        # Untuk Linear layer, kita akan menggunakan self.quant_group_size sebagai argumen 'group_size'
        # di make_l4q_linear untuk konsistensi nama internal di L4QQuantizedLinear.

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
                # Teruskan l4q_params ke ResnetBlock
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         l4q_params=self.l4q_params)) # <<< Teruskan l4q_params
                block_in = block_out
                if curr_res in attn_resolutions:
                    # Teruskan l4q_params ke make_attn -> AttnBlock
                    attn.append(make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params)) # <<< Teruskan l4q_params
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                # Teruskan l4q_params ke Downsample
                down.downsample = Downsample(block_in, resamp_with_conv, l4q_params=self.l4q_params) # <<< Teruskan l4q_params
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        # Teruskan l4q_params ke ResnetBlock dan make_attn
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level] # Ini akan menjadi block_in untuk iterasi pertama di level ini
            
            # block_in untuk ResNetBlock di upsampling adalah kombinasi dari output level sebelumnya 
            # dan skip connection. Nilai awalnya adalah output dari middle block (block_in terakhir dari downsampling).
            current_block_in_upsampling = block_in # Ini adalah block_in dari akhir down/middle path

            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    # Pada iterasi terakhir, skip_in adalah dari resolusi yang sesuai di down path
                    skip_in_channels_actual = ch*in_ch_mult[i_level] 
                else:
                    # Pada iterasi lainnya, skip_in adalah output dari ResNetBlock sebelumnya di level upsampling yang sama
                    # (kecuali untuk i_block=0 dimana skip_in adalah dari down path juga)
                    skip_in_channels_actual = ch*ch_mult[i_level]


                # Input ke ResnetBlock adalah (output h dari iterasi/level sebelumnya) + (skip connection)
                # Ukuran h (current_block_in_upsampling) harus sama dengan block_out dari iterasi sebelumnya
                # atau output dari upsample jika i_block = 0 dan bukan level pertama upsampling.
                # Ukuran skip connection (skip_in_channels_actual) diambil dari hs.
                
                # Perlu lebih hati-hati dengan `block_in` yang digunakan untuk `ResnetBlock` di sini.
                # `block_in` yang di pass ke `ResnetBlock` adalah `current_block_in_upsampling + skip_in_channels_actual`
                # `block_in` untuk iterasi selanjutnya akan menjadi `block_out` dari ResnetBlock saat ini.

                # Menggunakan nama variabel yang lebih jelas untuk input ke ResNet
                input_to_resnet = current_block_in_upsampling + skip_in_channels_actual

                block.append(ResnetBlock(in_channels=input_to_resnet, # Ini adalah channel gabungan
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         l4q_params=self.l4q_params))
                current_block_in_upsampling = block_out # Output dari ResnetBlock menjadi input (non-skip part) untuk selanjutnya
                if curr_res in attn_resolutions:
                    attn.append(make_attn(current_block_in_upsampling, attn_type=attn_type, l4q_params=self.l4q_params))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # Upsample output dari level sebelumnya (current_block_in_upsampling)
                up.upsample = Upsample(current_block_in_upsampling, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res * 2
            self.up.insert(0, up)
            # Input untuk level upsampling berikutnya (current_block_in_upsampling) adalah output dari upsample,
            # atau output dari ResnetBlock terakhir jika tidak ada upsample (i_level == 0).
            # Ini ditangani oleh `h` di forward pass. `block_in` di sini sebenarnya adalah `h` di forward.
            block_in = current_block_in_upsampling # Ini untuk loop berikutnya `for i_level` jika diperlukan


        self.norm_out = Normalize(block_out) # block_out dari level terakhir upsampling
        if self.l4q_enabled:
            self.conv_out = make_l4q_conv2d(block_out, out_ch, kernel_size=3, stride=1, padding=1,
                                            lora_rank=self.lora_rank, n_bits=self.n_bits,
                                            alpha=self.alpha, quant_group_size=self.quant_group_size)
        else:
            self.conv_out = nn.Conv2d(block_out, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t=None, context=None): # `context` di sini adalah conditioning, bukan context cross-attn internal
        # Konteks untuk cross-attention biasanya disuntikkan ke blok Transformer, bukan digabung di awal.
        # Jika `conditioning_key: crossattn` digunakan, `context` akan diteruskan ke Transformer.
        # Jika `conditioning_key: concat`, maka x digabung dengan context.
        # Kode asli Anda sudah menangani `context` dengan `torch.cat` jika ada.
        # Saya akan biarkan logika ini untuk sekarang, tetapi perhatikan bagaimana context dipakai.
        # Jika `context` adalah untuk cross-attention, ia tidak seharusnya digabung di sini.
        # Jika model Unet ini dari OpenAIMode, ia mungkin tidak secara langsung menangani context untuk transformer.
        # Itu biasanya dilakukan oleh `SpatialTransformer`.

        # assert x.shape[2] == x.shape[3] == self.resolution # Periksa dimensi input latent
        
        # Handle context jika ada (misalnya untuk concat conditioning)
        # Perlu diperjelas apakah 'context' di sini adalah conditioning spasial atau untuk cross-attn
        # Kode asli Anda memiliki ini, jadi saya pertahankan. Jika ini UnetModel dari LDM, ia mungkin tidak
        # menangani 'context' seperti ini secara langsung.
        # if context is not None:
        #     # assume aligned context, cat along channel axis
        #     x = torch.cat((x, context), dim=1) # Ini akan mengubah in_channels efektif

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
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            # h adalah output dari level sebelumnya (atau middle block)
            for i_block in range(self.num_res_blocks+1):
                # Ambil skip connection yang sesuai
                skip_h = hs.pop()
                # Gabungkan h dengan skip_h
                h_cat = torch.cat([h, skip_h], dim=1)
                h = self.up[i_level].block[i_block](h_cat, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        # Mengembalikan bobot dari layer conv_out (bisa L4Q atau nn.Conv2d)
        if hasattr(self.conv_out, 'w0'): # Jika L4Q, kembalikan w0 (bobot asli)
            return self.conv_out.w0 
        else: # Jika nn.Conv2d standar
            return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = make_l4q_conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_l4q_conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = make_l4q_conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_l4q_conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([make_l4q_conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     make_l4q_conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = make_l4q_conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_l4q_conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = make_l4q_conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = make_l4q_conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = make_l4q_conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

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
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = make_l4q_conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z
