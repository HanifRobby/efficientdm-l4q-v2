import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math # Pastikan math diimpor

# --- Impor dari l4q_quant_core.py ---
from .l4q_quant_core import l4q_init_scale, get_quantization_bounds 

class L4QQuantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, lora_a, lora_b, alpha, n_bits, q_scale, group_size):
        # ... (kode forward tetap sama seperti yang Anda implementasikan) ...
        w_lora = alpha * (lora_b @ lora_a) 
        w_comb = w0 + w_lora 

        if group_size != -1 and w_comb.numel() > group_size and w_comb.numel() % group_size == 0 :
            original_shape = w_comb.shape
            num_groups_w = w_comb.numel() // group_size
            w_comb_grouped = w_comb.reshape(num_groups_w, group_size) # Menggunakan reshape
            
            if q_scale.numel() == num_groups_w:
                scale_expanded = q_scale.unsqueeze(1) 
            elif q_scale.numel() == 1: 
                scale_expanded = q_scale 
            else:
                raise ValueError(f"Shape q_scale ({q_scale.shape}) tidak cocok untuk group-wise quantization ({num_groups_w} groups).")
            
            w_scaled_grouped = w_comb_grouped / (scale_expanded + 1e-9) # Tambah epsilon
        else: 
            w_scaled = w_comb / (q_scale + 1e-9) # Tambah epsilon
        
        q_n, q_p = get_quantization_bounds(n_bits)
        
        if group_size != -1 and w_comb.numel() > group_size and w_comb.numel() % group_size == 0:
            quantized_w_int_grouped = torch.round(torch.clamp(w_scaled_grouped, q_n, q_p))
            w_q_grouped = quantized_w_int_grouped * scale_expanded
            w_q = w_q_grouped.reshape(original_shape) 
            w_scaled_for_ste = w_scaled_grouped 
            quantized_w_int_for_backward = quantized_w_int_grouped # Sudah grouped
        else:
            quantized_w_int = torch.round(torch.clamp(w_scaled, q_n, q_p))
            w_q = quantized_w_int * q_scale 
            w_scaled_for_ste = w_scaled
            quantized_w_int_for_backward = quantized_w_int # Per-tensor

        output = F.linear(x, w_q) 
        
        ctx.save_for_backward(x, w0, lora_a, lora_b, q_scale, w_q, quantized_w_int_for_backward, w_comb)
        ctx.alpha = alpha
        ctx.n_bits = n_bits
        ctx.group_size = group_size # Simpan group_size aktual yang digunakan
        ctx.q_n = q_n
        ctx.q_p = q_p
        ctx.w_scaled_for_ste = w_scaled_for_ste
        ctx.effective_group_size = group_size if (group_size != -1 and w_comb.numel() > group_size and w_comb.numel() % group_size == 0) else -1


        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w0, lora_a, lora_b, q_scale, w_q, quantized_w_int, w_comb = ctx.saved_tensors
        alpha = ctx.alpha
        # n_bits = ctx.n_bits # Tidak dipakai di backward
        effective_group_size = ctx.effective_group_size # Gunakan effective_group_size
        q_n = ctx.q_n
        q_p = ctx.q_p
        w_scaled_for_ste = ctx.w_scaled_for_ste

        grad_x = grad_w0 = grad_lora_a = grad_lora_b = grad_q_scale = None
        # grad_alpha, grad_n_bits, grad_group_size_param tidak ada

        if ctx.needs_input_grad[0]: 
            grad_x = F.linear(grad_output, w_q.transpose(0, 1))

        grad_L_wrt_wq = grad_output.transpose(-2, -1) @ x 
        if len(grad_L_wrt_wq.shape) > 2: 
            grad_L_wrt_wq = grad_L_wrt_wq.sum(dim=0)
        
        if effective_group_size != -1:
            # w_scaled_for_ste sudah grouped
            ste_mask_grouped = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
            ste_mask = ste_mask_grouped.reshape(w_comb.shape)
        else:
            ste_mask = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
        
        grad_L_wrt_wq_masked_by_ste = grad_L_wrt_wq * ste_mask

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]: 
            if ctx.needs_input_grad[2]: 
                grad_lora_a = alpha * (lora_b.transpose(0,1) @ grad_L_wrt_wq_masked_by_ste)
            if ctx.needs_input_grad[3]: 
                grad_lora_b = (grad_L_wrt_wq_masked_by_ste @ lora_a.transpose(0,1)) * alpha
        
        if ctx.needs_input_grad[6]: # q_scale adalah argumen ke-7 (index 6)
            if effective_group_size != -1:
                num_groups_w = w_comb.numel() // effective_group_size
                # quantized_w_int sudah grouped
                grad_L_wrt_wq_grouped = grad_L_wrt_wq.reshape(num_groups_w, effective_group_size)
                # quantized_w_int_grouped = quantized_w_int.reshape(num_groups_w, effective_group_size) # Tidak perlu reshape jika sudah grouped
                
                grad_q_scale_grouped = (grad_L_wrt_wq_grouped * quantized_w_int).sum(dim=1) # quantized_w_int sudah grouped
                grad_q_scale = grad_q_scale_grouped
            else: 
                grad_q_scale = (grad_L_wrt_wq * quantized_w_int).sum()
        
        return grad_x, grad_w0, grad_lora_a, grad_lora_b, None, None, grad_q_scale, None # None untuk alpha, n_bits, group_size

class L4QQuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 lora_rank: int, n_bits: int, alpha: float = 1.0, 
                 group_size: int = -1, 
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.n_bits = n_bits
        self.alpha = alpha
        self.group_size = group_size # group_size yang diinginkan

        self.w0 = Parameter(torch.Tensor(out_features, in_features))
        self.lora_b = Parameter(torch.Tensor(out_features, lora_rank))
        self.lora_a = Parameter(torch.Tensor(lora_rank, in_features))
        
        num_elements = out_features * in_features
        # Tentukan effective_group_size untuk inisialisasi q_scale
        self.effective_group_size_init = self.group_size
        if self.group_size != -1:
            if num_elements == 0: # Handle tensor kosong
                 self.effective_group_size_init = -1 
            elif num_elements < self.group_size or num_elements % self.group_size != 0 :
                print(f"L4QLinear INFO: num_elements ({num_elements}) or divisibility issue with group_size ({self.group_size}). Using per-tensor for q_scale init.")
                self.effective_group_size_init = -1 # Fallback ke per-tensor untuk q_scale jika tidak bisa dibagi atau terlalu kecil

        if self.effective_group_size_init != -1:
            num_groups = num_elements // self.effective_group_size_init
            self.q_scale = Parameter(torch.Tensor(num_groups))
        else:
            self.q_scale = Parameter(torch.Tensor(1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.w0.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w0)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.bias)

        with torch.no_grad():
            w_lora_init = self.alpha * (self.lora_b @ self.lora_a)
            w_comb_init = self.w0 + w_lora_init
            
            # Gunakan effective_group_size_init untuk memanggil l4q_init_scale
            initial_scale = l4q_init_scale(w_comb_init, self.n_bits, self.effective_group_size_init)
            
            if self.effective_group_size_init != -1 and initial_scale.numel() == self.q_scale.numel():
                 self.q_scale.data.copy_(initial_scale) # Tidak perlu squeeze jika l4q_init_scale mengembalikan [num_groups]
            elif initial_scale.numel() == 1: # Per-tensor atau fallback
                 self.q_scale.data.fill_(initial_scale.item())
            else:
                print(f"WARNING L4QLinear: Mismatch in q_scale initialization. initial_scale shape: {initial_scale.shape}, self.q_scale shape: {self.q_scale.shape}. Filling with first element.")
                self.q_scale.data.fill_(initial_scale[0].item() if initial_scale.numel() > 0 else 1e-9)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # group_size yang diteruskan ke fungsi adalah group_size yang diinginkan
        output = L4QQuantizedLinearFunction.apply(x, self.w0, self.lora_a, self.lora_b, 
                                                  self.alpha, self.n_bits, self.q_scale,
                                                  self.group_size) # Teruskan self.group_size asli
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'lora_rank={self.lora_rank}, n_bits={self.n_bits}, alpha={self.alpha}, '
                f'group_size={self.group_size}, bias={self.bias is not None}, '
                f'q_scale_shape={self.q_scale.shape}')
