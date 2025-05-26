# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
# LinearAttention akan diimpor dari ldm.modules.attention yang sudah dimodifikasi
from ldm.modules.attention import LinearAttention 

# --- Impor Fungsi Helper L4Q dari l4q_utils ---
try:
    # Menggunakan ..l4q karena model.py ada di ldm/modules/diffusionmodules/
    from ..l4q.l4q_utils import make_l4q_linear, make_l4q_conv2d
except ImportError:
    print("PERINGATAN KRITIKAL di model.py: Gagal mengimpor helper L4Q dari ..l4q.l4q_utils. Layer L4Q TIDAK AKAN BERFUNGSI.")
    # Fallback placeholder jika impor gagal
    def make_l4q_linear(in_features, out_features, bias=True, l4q_enabled_passed=False, **kwargs):
        print(f"model.py: Fallback ke nn.Linear ({in_features}, {out_features}) karena impor helper L4Q gagal.")
        return nn.Linear(in_features, out_features, bias=bias)
    def make_l4q_conv2d(dims, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, l4q_enabled_passed=False, **kwargs):
        print(f"model.py: Fallback ke nn.Conv{dims}d ({in_channels}, {out_channels}) karena impor helper L4Q gagal.")
        if dims == 1: return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if dims == 2: return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if dims == 3: return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        raise ValueError(f"Dimensi tidak didukung: {dims}")

# --- Fungsi Utilitas Model (tidak berubah) ---
def get_timestep_embedding(timesteps, embedding_dim):
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
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    # Pastikan num_groups valid
    if in_channels == 0: # Handle kasus in_channels adalah 0 jika mungkin terjadi
        return nn.Identity()
    if num_groups > in_channels:
        num_groups = 1 # Atau in_channels jika Anda ingin grup per channel
    if in_channels % num_groups != 0:
        # Cari pembagi terbesar dari in_channels yang <= num_groups default
        for i in range(min(num_groups, in_channels), 0, -1):
            if in_channels % i == 0:
                num_groups = i
                break
        if in_channels % num_groups != 0: # Jika masih tidak bisa dibagi (misalnya in_channels adalah prima)
            num_groups = 1 
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# --- Sub-Modul dengan Integrasi L4Q ---
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, l4q_params=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
            self.conv = make_l4q_conv2d(2, # dims=2
                                        in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                        l4q_enabled_passed=l4q_enabled,
                                        lora_rank=l4q_params.get("lora_rank", 4) if l4q_params else 4,
                                        n_bits=l4q_params.get("n_bits", 4) if l4q_params else 4,
                                        alpha=l4q_params.get("alpha", 1.0) if l4q_params else 1.0,
                                        quant_group_size=l4q_params.get("quant_group_size", -1) if l4q_params else -1)
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
            self.conv = make_l4q_conv2d(2, # dims=2
                                        in_channels, in_channels, kernel_size=3, stride=2, padding=0,
                                        l4q_enabled_passed=l4q_enabled,
                                        lora_rank=l4q_params.get("lora_rank", 4) if l4q_params else 4,
                                        n_bits=l4q_params.get("n_bits", 4) if l4q_params else 4,
                                        alpha=l4q_params.get("alpha", 1.0) if l4q_params else 1.0,
                                        quant_group_size=l4q_params.get("quant_group_size", -1) if l4q_params else -1)
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
        self.conv1 = make_l4q_conv2d(2, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                     l4q_enabled_passed=l4q_enabled,
                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        
        if temb_channels > 0:
            self.temb_proj = make_l4q_linear(temb_channels, out_channels, l4q_enabled_passed=l4q_enabled,
                                             lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        else:
            self.temb_proj = None # Atau nn.Identity() jika forward pass memerlukannya
        
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_l4q_conv2d(2, out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                     l4q_enabled_passed=l4q_enabled,
                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_l4q_conv2d(2, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                                     l4q_enabled_passed=l4q_enabled,
                                                     lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
            else:
                self.nin_shortcut = make_l4q_conv2d(2, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                                    l4q_enabled_passed=l4q_enabled,
                                                    lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
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

class LinAttnBlock(LinearAttention): # LinearAttention sudah dimodifikasi di attention.py
    def __init__(self, in_channels, l4q_params=None): # Menerima l4q_params
        # LinearAttention di attention.py akan menangani l4q_params
        # Pastikan argumen yang diteruskan ke super() sesuai dengan __init__ LinearAttention
        # Jika LinearAttention di attention.py menggunakan 'dim' bukan 'in_channels':
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels, l4q_params=l4q_params)

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
        self.q = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                 l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.k = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                 l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.v = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                 l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.proj_out = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
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
        v = v.reshape(b,c,h*w).permute(0,2,1)
        h_ = torch.bmm(w_,v)
        h_ = h_.permute(0,2,1).reshape(b,c,h,w)
        h_ = self.proj_out(h_)
        return x+h_

def make_attn(in_channels, attn_type="vanilla", l4q_params=None):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels, l4q_params=l4q_params)
    elif attn_type == "none":
        return nn.Identity()
    else: # linear attention
        return LinAttnBlock(in_channels, l4q_params=l4q_params)

class Model(nn.Module): # UNetModel
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

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        self.use_timestep = use_timestep
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                make_l4q_linear(self.ch, self.temb_ch, l4q_enabled_passed=l4q_enabled,
                                lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs),
                make_l4q_linear(self.temb_ch, self.temb_ch, l4q_enabled_passed=l4q_enabled,
                                lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs),
            ])
        
        self.conv_in = make_l4q_conv2d(2, in_channels, self.ch, kernel_size=3, stride=1, padding=1,
                                       l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b,
                                       alpha=alph, quant_group_size=q_gs)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in_for_next_level = self.ch # Inisialisasi
        for i_level in range(self.num_resolutions):
            block_list_modules = nn.ModuleList() # Ganti nama variabel
            attn_list_modules = nn.ModuleList()  # Ganti nama variabel
            block_in = self.ch*in_ch_mult[i_level]
            block_out = self.ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks): # i_block tidak dipakai
                block_list_modules.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                                     temb_channels=self.temb_ch, dropout=dropout,
                                                     l4q_params=self.l4q_params))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_list_modules.append(make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params))
            down_module = nn.Module()
            down_module.block = block_list_modules
            down_module.attn = attn_list_modules
            if i_level != self.num_resolutions-1:
                down_module.downsample = Downsample(block_in, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res // 2
            self.down.append(down_module)
            block_in_for_next_level = block_in # Simpan block_in terakhir untuk mid block

        self.mid = nn.Module()
        mid_channels = block_in_for_next_level
        self.mid.block_1 = ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(mid_channels, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        self.up = nn.ModuleList()
        current_block_in_upsampling = mid_channels # Output dari middle block
        for i_level in reversed(range(self.num_resolutions)):
            block_list_modules = nn.ModuleList()
            attn_list_modules = nn.ModuleList()
            block_out = self.ch*ch_mult[i_level]
            
            for i_block_up in range(self.num_res_blocks+1): # Ganti nama variabel
                # Tentukan channel untuk skip connection dari `in_ch_mult`
                # Skip connection berasal dari resolusi yang sama di down path
                skip_in_channels_actual = self.ch*in_ch_mult[i_level if i_block_up == self.num_res_blocks else i_level] # Ini perlu dicek logikanya
                # Lebih tepatnya, skip_in_channels_actual adalah ch*in_ch_mult[i_level] untuk semua blok di level ini
                # atau ch*ch_mult[i_level] jika skip diambil dari output ResBlock sebelumnya di up path.
                # Kode asli LDM lebih kompleks dalam menangani skip, mari sederhanakan untuk sekarang:
                # Asumsikan skip_in adalah output dari down block yang sesuai.
                # hs.pop() di forward akan memberikan skip connection yang benar.
                # Jadi, input_to_resnet akan memiliki channel = current_block_in_upsampling + channel_dari_hs_pop

                # Channel dari hs.pop() akan menjadi ch*in_ch_mult[i_level] atau ch*ch_mult[i_level]
                # tergantung pada apakah itu output conv_in atau output ResBlock/Downsample.
                # Untuk upsampling, input ke ResBlock adalah (h dari level atas) + (skip dari hs)
                # Ukuran `h` adalah `current_block_in_upsampling`. Ukuran `skip` akan bervariasi.
                # Kita perlu tahu channel dari `hs.pop()` di forward pass.
                # Mari asumsikan `input_to_resnet` akan dihandle di forward pass dengan concat.
                # `ResnetBlock` perlu tahu channel input gabungannya.
                # Ini bagian yang rumit. Kode asli LDM menghitung `block_in` untuk `ResnetBlock` di upsampling
                # berdasarkan `ch*ch_mult[i_level]` (untuk skip) + `ch*ch_mult[level_atas]` (untuk `h`).

                # Mari kita sederhanakan: ResnetBlock akan menerima channel gabungan.
                # Channel `h` (current_block_in_upsampling) + channel `skip_h` (dari hs.pop()).
                # Channel `skip_h` adalah `ch*ch_mult[i_level]` (jika i_block < num_res_blocks)
                # atau `ch*in_ch_mult[i_level]` (jika i_block == num_res_blocks, skip dari awal level down).
                # Ini masih perlu disesuaikan dengan bagaimana `hs` di-populate dan di-pop.
                # Untuk sekarang, asumsikan `input_to_resnet` adalah placeholder.
                # Di forward, kita akan tahu channel sebenarnya saat concat.
                # Namun, __init__ butuh channel.
                
                # Mengikuti logika LDM:
                # `block_in` untuk ResBlock di upsampling adalah `ch*ch_mult[i_level]` (dari `h` setelah upsample/ResBlock sebelumnya di up)
                # ditambah `ch*in_ch_mult[i_level]` (dari skip connection `hs`).
                # Ini berarti ResBlock harus diinisialisasi dengan `current_block_in_upsampling + ch*in_ch_mult[i_level]`
                # atau `current_block_in_upsampling + ch*ch_mult[i_level]` tergantung `i_block_up`.

                # Mari kita gunakan pendekatan yang lebih sederhana untuk __init__ dan biarkan forward pass menangani concat.
                # `ResnetBlock` akan diinisialisasi dengan channel input yang diharapkan setelah concat.
                # Channel `h` adalah `current_block_in_upsampling`. Channel skip dari `hs` adalah `ch*ch_mult[i_level]`.
                # Jadi input ke ResNet adalah `current_block_in_upsampling + ch*ch_mult[i_level]`
                
                # Logika asli LDM untuk `block_in` di upsampling:
                # if i_level == self.num_resolutions - 1: # Level terendah up, skip dari mid
                #     input_concat_ch = ch * ch_mult[i_level] # dari hs
                # else: # Skip dari level yang sesuai di down path
                #     input_concat_ch = ch * in_ch_mult[i_level+1] # ini tidak tepat

                # Mari kita gunakan channel dari hs yang sesuai dengan level ini.
                # hs[-1] sebelum pop di forward akan memiliki channel `ch*ch_mult[i_level]` (jika dari ResBlock)
                # atau `ch*in_ch_mult[i_level]` (jika dari conv_in atau downsample pertama).
                # Ini perlu konsisten dengan apa yang disimpan di `hs`.
                # Asumsikan `hs` berisi output dari setiap blok di down path.
                # `hs.pop()` akan memberikan skip connection. Channelnya adalah `ch*ch_mult[i_level]`
                # (output dari ResBlock di down path pada resolusi yang sama).

                input_channels_for_resnet = current_block_in_upsampling + ch*ch_mult[i_level]

                block_list_modules.append(ResnetBlock(in_channels=input_channels_for_resnet, out_channels=block_out,
                                                     temb_channels=self.temb_ch, dropout=dropout,
                                                     l4q_params=self.l4q_params))
                current_block_in_upsampling = block_out
                if curr_res in attn_resolutions:
                    attn_list_modules.append(make_attn(current_block_in_upsampling, attn_type=attn_type, l4q_params=self.l4q_params))
            
            up_module = nn.Module()
            up_module.block = block_list_modules
            up_module.attn = attn_list_modules
            if i_level != 0:
                up_module.upsample = Upsample(current_block_in_upsampling, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res * 2
            self.up.insert(0, up_module)
            # `block_in` untuk level berikutnya tidak secara eksplisit dihitung di sini, karena `h` di forward pass
            # akan menjadi `current_block_in_upsampling` setelah upsample (jika ada).

        self.norm_out = Normalize(current_block_in_upsampling) # Ini adalah block_out dari level terakhir upsampling
        self.conv_out = make_l4q_conv2d(2, current_block_in_upsampling, out_ch, kernel_size=3, stride=1, padding=1,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b,
                                        alpha=alph, quant_group_size=q_gs)
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
                h_ = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    if i_block < len(self.down[i_level].attn): # Pastikan index valid
                         h_ = self.down[i_level].attn[i_block](h_)
                hs.append(h_)
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
                if len(self.up[i_level].attn) > 0:
                    if i_block < len(self.up[i_level].attn): # Pastikan index valid
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

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 l4q_params=None, **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        self.conv_in = make_l4q_conv2d(2, in_channels, self.ch, kernel_size=3, stride=1, padding=1,
                                       l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in_for_next_level = self.ch
        for i_level in range(self.num_resolutions):
            block_list_modules = nn.ModuleList()
            attn_list_modules = nn.ModuleList()
            block_in = self.ch*in_ch_mult[i_level]
            block_out = self.ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block_list_modules.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                                     temb_channels=self.temb_ch, dropout=dropout,
                                                     l4q_params=self.l4q_params))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn_list_modules.append(make_attn(block_in, attn_type=attn_type, l4q_params=self.l4q_params))
            down_module = nn.Module()
            down_module.block = block_list_modules
            down_module.attn = attn_list_modules
            if i_level != self.num_resolutions-1:
                down_module.downsample = Downsample(block_in, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res // 2
            self.down.append(down_module)
            block_in_for_next_level = block_in

        self.mid = nn.Module()
        mid_channels = block_in_for_next_level
        self.mid.block_1 = ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(mid_channels, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        self.norm_out = Normalize(mid_channels)
        self.conv_out = make_l4q_conv2d(2, mid_channels, 2*z_channels if double_z else z_channels,
                                        kernel_size=3, stride=1, padding=1,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h_ = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                     if i_block < len(self.down[i_level].attn):
                        h_ = self.down[i_level].attn[i_block](h_)
                hs.append(h_)
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
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", l4q_params=None, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.l4q_params = l4q_params

        l4q_enabled = self.l4q_params.get("enabled", False) if self.l4q_params else False
        lora_r = self.l4q_params.get("lora_rank", 4) if self.l4q_params else 4
        n_b = self.l4q_params.get("n_bits", 4) if self.l4q_params else 4
        alph = self.l4q_params.get("alpha", 1.0) if self.l4q_params else 1.0
        q_gs = self.l4q_params.get("quant_group_size", -1) if self.l4q_params else -1

        block_in_initial = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        
        self.conv_in = make_l4q_conv2d(2, z_channels, block_in_initial, kernel_size=3, stride=1, padding=1,
                                       l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in_initial, out_channels=block_in_initial,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)
        self.mid.attn_1 = make_attn(block_in_initial, attn_type=attn_type, l4q_params=self.l4q_params)
        self.mid.block_2 = ResnetBlock(in_channels=block_in_initial, out_channels=block_in_initial,
                                       temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params)

        self.up = nn.ModuleList()
        current_block_in_upsampling = block_in_initial
        for i_level in reversed(range(self.num_resolutions)):
            block_list_modules = nn.ModuleList()
            attn_list_modules = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks+1): # i_block tidak dipakai
                block_list_modules.append(ResnetBlock(in_channels=current_block_in_upsampling, out_channels=block_out,
                                                     temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params))
                current_block_in_upsampling = block_out
                if curr_res in attn_resolutions:
                    attn_list_modules.append(make_attn(current_block_in_upsampling, attn_type=attn_type, l4q_params=self.l4q_params))
            up_module = nn.Module()
            up_module.block = block_list_modules
            up_module.attn = attn_list_modules
            if i_level != 0:
                up_module.upsample = Upsample(current_block_in_upsampling, resamp_with_conv, l4q_params=self.l4q_params)
                curr_res = curr_res * 2
            self.up.insert(0, up_module)

        self.norm_out = Normalize(current_block_in_upsampling)
        self.conv_out = make_l4q_conv2d(2, current_block_in_upsampling, out_ch, kernel_size=3, stride=1, padding=1,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
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
                    if i_block < len(self.up[i_level].attn):
                        h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end: return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out: h = torch.tanh(h)
        return h

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, l4q_params=None, *args, **kwargs):
        super().__init__()
        self.l4q_params = l4q_params
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        conv1 = make_l4q_conv2d(2, in_channels, in_channels, 1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        conv2 = make_l4q_conv2d(2, 2*in_channels, in_channels, 1, l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.conv_out = make_l4q_conv2d(2, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)

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
    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]: x = layer(x, None)
            else: x = layer(x)
        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x

class UpsampleDecoder(nn.Module): # Mirip SimpleDecoder, tapi dengan parameterisasi berbeda
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0, l4q_params=None):
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
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block_level = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block_level.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                                 temb_channels=self.temb_ch, dropout=dropout, l4q_params=self.l4q_params))
                block_in = block_out
            self.res_blocks.append(res_block_level)
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True, l4q_params=self.l4q_params))
        self.norm_out = Normalize(block_in)
        self.conv_out = make_l4q_conv2d(2, block_in, out_channels, kernel_size=3, stride=1, padding=1,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x):
        h = x
        for k in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[k][i_block](h, None)
            if k != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2, l4q_params=None):
        super().__init__()
        self.factor = factor
        self.l4q_params = l4q_params
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.conv_in = make_l4q_conv2d(2, in_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                                       l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                                     temb_channels=0, dropout=0.0, l4q_params=self.l4q_params) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels, l4q_params=self.l4q_params)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels, out_channels=mid_channels,
                                                     temb_channels=0, dropout=0.0, l4q_params=self.l4q_params) for _ in range(depth)])
        self.conv_out = make_l4q_conv2d(2, mid_channels, out_channels, kernel_size=1,
                                        l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1: x = block(x, None)
        if self.factor != 1.0:
            x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2: x = block(x, None)
        x = self.conv_out(x)
        return x

class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1, l4q_params=None):
        super().__init__()
        self.l4q_params = l4q_params
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None, l4q_params=self.l4q_params) # out_ch=None karena ini VAE encoder
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth,
                                       l4q_params=self.l4q_params)
    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x

class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1, l4q_params=None):
        super().__init__()
        self.l4q_params = l4q_params
        rescaler_input_channels = z_channels # Input ke rescaler
        rescaler_output_channels = z_channels # Asumsi rescaler tidak mengubah channel utama, atau sesuaikan
        # Jika rescaler mengubah channel, decoder_input_z_channels = output channel rescaler
        
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=rescaler_input_channels, 
                                       mid_channels=rescaler_output_channels, # Atau mid_channels yang berbeda
                                       out_channels=rescaler_output_channels, depth=rescale_module_depth,
                                       l4q_params=self.l4q_params)
        self.decoder = Decoder(out_ch=out_ch, z_channels=rescaler_output_channels, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch, l4q_params=self.l4q_params)
    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x

class Upsampler(nn.Module): # Digunakan oleh image_editing.PaintByExample
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2, l4q_params=None):
        super().__init__()
        self.l4q_params = l4q_params
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1 if out_size > in_size else 1
        factor_up = float(out_size) / in_size
        
        rescaler_mid_channels = in_channels * ch_mult if factor_up > 1.0 and ch_mult > 0 else in_channels
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, 
                                       mid_channels=rescaler_mid_channels,
                                       out_channels=in_channels, depth=1, 
                                       l4q_params=self.l4q_params)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels, 
                               ch_mult=[ch_mult for _ in range(num_blocks)], l4q_params=self.l4q_params)
    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x

class Resize(nn.Module): 
    def __init__(self, in_channels=None, learned=False, mode="bilinear", l4q_params=None):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        self.l4q_params = l4q_params
        if self.with_conv:
            raise NotImplementedError("Learned Resize with L4Q not fully implemented/verified here.")
    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0: return x
        else: return torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)

class FirstStagePostProcessor(nn.Module):
    def __init__(self, ch_mult:list, in_channels, # in_channels adalah output dari VAE (z_fs)
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None, # Channel setelah proj, input ke ResNet pertama
                 dropout=0.,
                 pretrained_config=None,
                 l4q_params=None):
        super().__init__()
        self.l4q_params = l4q_params
        if pretrained_config is None:
            assert pretrained_model is not None
            self.pretrained_model = pretrained_model
        else:
            self.instantiate_pretrained(pretrained_config)
        self.do_reshape = reshape
        if n_channels is None:
            try: n_channels = self.pretrained_model.encoder.ch 
            except AttributeError: n_channels = in_channels

        self.proj_norm = Normalize(in_channels,num_groups=max(1, in_channels//2 if in_channels >=2 else 1))
        
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.proj = make_l4q_conv2d(2, in_channels,n_channels,kernel_size=3, stride=1,padding=1,
                                    l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        blocks = []
        downs = []
        ch_in = n_channels
        for m_idx, m_val in enumerate(ch_mult): # Ganti nama variabel
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m_val*n_channels,dropout=dropout, temb_channels=0, l4q_params=self.l4q_params))
            ch_in = m_val * n_channels
            if m_idx < len(ch_mult) -1 :
                downs.append(Downsample(ch_in, with_conv=False, l4q_params=self.l4q_params))
            else: 
                downs.append(nn.Identity())
        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)

    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        for param in self.pretrained_model.parameters(): param.requires_grad = False

    @torch.no_grad()
    def encode_with_pretrained(self,x):
        if hasattr(self.pretrained_model, 'encode') and callable(self.pretrained_model.encode):
            c = self.pretrained_model.encode(x)
            if hasattr(c, 'mode') and callable(c.mode): 
                c = c.mode()
            return c
        else: 
            raise NotImplementedError("Pretrained model missing 'encode' or 'DiagonalGaussianDistribution'.")

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)
        for i in range(len(self.model)):
            z = self.model[i](z,temb=None)
            z = self.downsampler[i](z)
        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z
