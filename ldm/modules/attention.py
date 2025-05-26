from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint # checkpoint dari util OpenAI UNet

# --- Impor Fungsi Helper L4Q dari l4q_utils ---
try:
    # Menggunakan .l4q karena attention.py ada di ldm/modules/
    from .l4q.l4q_utils import make_l4q_linear, make_l4q_conv2d
except ImportError:
    print("PERINGATAN KRITIKAL di attention.py: Gagal mengimpor helper L4Q dari .l4q.l4q_utils. Layer L4Q TIDAK AKAN BERFUNGSI.")
    # Fallback placeholder jika impor gagal
    def make_l4q_linear(in_features, out_features, bias=True, l4q_enabled_passed=False, **kwargs):
        print(f"attention.py: Fallback ke nn.Linear ({in_features}, {out_features}) karena impor helper L4Q gagal.")
        return nn.Linear(in_features, out_features, bias=bias)
    def make_l4q_conv2d(dims, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, l4q_enabled_passed=False, **kwargs):
        print(f"attention.py: Fallback ke nn.Conv{dims}d ({in_channels}, {out_channels}) karena impor helper L4Q gagal.")
        if dims == 1: return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if dims == 2: return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if dims == 3: return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        raise ValueError(f"Dimensi tidak didukung: {dims}")

# --- Fungsi Utilitas Standar (tidak berubah) ---
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# --- Modul dengan Integrasi L4Q ---
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, l4q_params=None):
        super().__init__()
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        self.proj = make_l4q_linear(dim_in, dim_out * 2, l4q_enabled_passed=l4q_enabled,
                                    lora_rank=l4q_params.get("lora_rank", 4) if l4q_params else 4,
                                    n_bits=l4q_params.get("n_bits", 4) if l4q_params else 4,
                                    alpha=l4q_params.get("alpha", 1.0) if l4q_params else 1.0,
                                    group_size=l4q_params.get("quant_group_size", -1) if l4q_params else -1)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., l4q_params=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        self.l4q_params = l4q_params 
        
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        if not glu:
            project_in_linear = make_l4q_linear(dim, inner_dim, l4q_enabled_passed=l4q_enabled,
                                                lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
            project_in = nn.Sequential(project_in_linear, nn.GELU())
        else:
            project_in = GEGLU(dim, inner_dim, l4q_params=self.l4q_params)

        output_linear = make_l4q_linear(inner_dim, dim_out, l4q_enabled_passed=l4q_enabled,
                                        lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), output_linear)
    def forward(self, x):
        return self.net(x)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels, num_groups=32): # Definisi lokal jika tidak diimpor dari model.py
    if in_channels == 0: return nn.Identity()
    if num_groups > in_channels: num_groups = 1
    if in_channels % num_groups != 0:
        for i in range(min(num_groups, in_channels), 0, -1):
            if in_channels % i == 0: num_groups = i; break
        if in_channels % num_groups != 0: num_groups = 1 
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, l4q_params=None):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.l4q_params = l4q_params
        
        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        # Ini adalah Conv2d dengan kernel_size=1, stride=1, padding=0
        self.to_qkv = make_l4q_conv2d(2, dim, hidden_dim * 3, 1, bias=False, l4q_enabled_passed=l4q_enabled,
                                      lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.to_out = make_l4q_conv2d(2, hidden_dim, dim, 1, l4q_enabled_passed=l4q_enabled,
                                      lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SpatialSelfAttention(nn.Module): # Mirip AttnBlock di model.py
    def __init__(self, in_channels, l4q_params=None):
        super().__init__()
        self.in_channels = in_channels
        self.l4q_params = l4q_params

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1
        
        self.norm = Normalize(in_channels)
        self.q = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, l4q_enabled_passed=l4q_enabled,
                                 lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.k = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, l4q_enabled_passed=l4q_enabled,
                                 lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.v = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, l4q_enabled_passed=l4q_enabled,
                                 lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.proj_out = make_l4q_conv2d(2, in_channels, in_channels, kernel_size=1, l4q_enabled_passed=l4q_enabled,
                                        lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1).contiguous()
        k = k.reshape(b,c,h*w).contiguous()
        w_ = torch.bmm(q,k) * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b,c,h*w).permute(0,2,1).contiguous()
        h_ = torch.bmm(w_,v)
        h_ = h_.permute(0,2,1).reshape(b,c,h,w).contiguous()
        h_ = self.proj_out(h_)
        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., l4q_params=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.l4q_params = l4q_params

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = make_l4q_linear(query_dim, inner_dim, bias=False, l4q_enabled_passed=l4q_enabled,
                                    lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        self.to_k = make_l4q_linear(context_dim, inner_dim, bias=False, l4q_enabled_passed=l4q_enabled,
                                    lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        self.to_v = make_l4q_linear(context_dim, inner_dim, bias=False, l4q_enabled_passed=l4q_enabled,
                                    lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        
        to_out_linear = make_l4q_linear(inner_dim, query_dim, l4q_enabled_passed=l4q_enabled,
                                        lora_rank=lora_r, n_bits=n_b, alpha=alph, group_size=q_gs)
        self.to_out = nn.Sequential(to_out_linear, nn.Dropout(dropout))
    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_val = max_neg_value(sim)
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_val)
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False, l4q_params=None):
        super().__init__()
        self.l4q_params = l4q_params
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, l4q_params=self.l4q_params)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, l4q_params=self.l4q_params)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, l4q_params=self.l4q_params)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, l4q_params=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.l4q_params = l4q_params

        l4q_enabled = l4q_params.get("enabled", False) if l4q_params else False
        lora_r = l4q_params.get("lora_rank", 4) if l4q_params else 4
        n_b = l4q_params.get("n_bits", 4) if l4q_params else 4
        alph = l4q_params.get("alpha", 1.0) if l4q_params else 1.0
        q_gs = l4q_params.get("quant_group_size", -1) if l4q_params else -1

        self.proj_in = make_l4q_conv2d(2, in_channels, inner_dim, kernel_size=1, stride=1, padding=0,
                                       l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, l4q_params=self.l4q_params)
                for _ in range(depth)]
        )
        proj_out_layer = make_l4q_conv2d(2, inner_dim, in_channels, kernel_size=1, stride=1, padding=0,
                                         l4q_enabled_passed=l4q_enabled, lora_rank=lora_r, n_bits=n_b, alpha=alph, quant_group_size=q_gs)
        self.proj_out = zero_module(proj_out_layer)
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
