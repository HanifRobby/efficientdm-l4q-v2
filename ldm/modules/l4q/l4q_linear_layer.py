import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from .l4q_quant_core import l4q_init_scale, get_quantization_bounds 

GRADCHECK_DTYPE = torch.double

def check_tensor_grad(grad_tensor, name="gradient", expected_dtype=GRADCHECK_DTYPE):
    if grad_tensor is None:
        print(f"  DEBUG GRAD: {name}: None (OK)")
        return True
    if not isinstance(grad_tensor, torch.Tensor):
        print(f"  DEBUG GRAD ERROR: {name}: Bukan Tensor! Tipe: {type(grad_tensor)}")
        return False
    valid = True
    print(f"  DEBUG GRAD: {name}: shape={grad_tensor.shape}, dtype={grad_tensor.dtype}, device={grad_tensor.device}, requires_grad={grad_tensor.requires_grad}")
    if torch.isnan(grad_tensor).any(): print(f"  DEBUG GRAD WARNING: {name} mengandung NaN!"); valid = False
    if torch.isinf(grad_tensor).any(): print(f"  DEBUG GRAD WARNING: {name} mengandung Inf!"); valid = False
    if grad_tensor.dtype != expected_dtype: print(f"  DEBUG GRAD WARNING: {name} dtype ({grad_tensor.dtype}) tidak cocok ({expected_dtype})!")
    return valid

class L4QQuantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, lora_a, lora_b, alpha, n_bits, q_scale, group_size):
        # ... (Kode forward Anda yang sudah ada, pastikan effective_group_size_fwd dihitung dengan benar) ...
        w_lora = alpha * (lora_b @ lora_a) 
        w_comb = w0 + w_lora 

        effective_group_size_fwd = group_size
        if group_size != -1:
            if w_comb.numel() == 0 or w_comb.numel() < group_size or w_comb.numel() % group_size != 0:
                effective_group_size_fwd = -1 

        w_scaled_grouped = None 
        w_scaled = None       

        if effective_group_size_fwd != -1:
            original_shape = w_comb.shape
            num_groups_w = w_comb.numel() // effective_group_size_fwd
            w_comb_grouped = w_comb.reshape(num_groups_w, effective_group_size_fwd)
            
            current_q_scale = q_scale
            if q_scale.numel() == num_groups_w:
                scale_expanded = current_q_scale.unsqueeze(1) 
            elif q_scale.numel() == 1: 
                scale_expanded = current_q_scale 
            else:
                scale_expanded = current_q_scale 
                effective_group_size_fwd = -1 
            
            if effective_group_size_fwd != -1:
                 w_scaled_grouped = w_comb_grouped / (scale_expanded + 1e-9)
        
        if effective_group_size_fwd == -1: 
            w_scaled = w_comb / (q_scale + 1e-9) if q_scale.numel() > 1 else w_comb / (q_scale.item() + 1e-9)

        q_n, q_p = get_quantization_bounds(n_bits)
        
        if effective_group_size_fwd != -1:
            quantized_w_int_grouped = torch.round(torch.clamp(w_scaled_grouped, q_n, q_p))
            w_q_grouped = quantized_w_int_grouped * scale_expanded
            w_q = w_q_grouped.reshape(original_shape) 
            w_scaled_for_ste = w_scaled_grouped 
            quantized_w_int_for_backward = quantized_w_int_grouped
        else:
            quantized_w_int = torch.round(torch.clamp(w_scaled, q_n, q_p))
            w_q = quantized_w_int * (q_scale if q_scale.numel() > 1 else q_scale.item())
            w_scaled_for_ste = w_scaled
            quantized_w_int_for_backward = quantized_w_int

        output = F.linear(x, w_q) 
        
        ctx.save_for_backward(x, w0, lora_a, lora_b, q_scale, w_q, quantized_w_int_for_backward, w_comb)
        ctx.alpha = alpha
        ctx.q_n = q_n
        ctx.q_p = q_p
        ctx.w_scaled_for_ste = w_scaled_for_ste
        ctx.effective_group_size_used_in_fwd = effective_group_size_fwd
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w0, lora_a, lora_b, q_scale, w_q, quantized_w_int, w_comb = ctx.saved_tensors
        alpha = ctx.alpha
        effective_group_size = ctx.effective_group_size_used_in_fwd 
        q_n = ctx.q_n
        q_p = ctx.q_p
        w_scaled_for_ste = ctx.w_scaled_for_ste

        grad_x = grad_w0 = grad_lora_a = grad_lora_b = grad_q_scale = None
        
        grad_output_dtype = grad_output.to(x.dtype)

        if ctx.needs_input_grad[0]: 
            grad_x = F.linear(grad_output_dtype, w_q.transpose(0, 1).to(x.dtype))

        grad_L_wrt_wq_intermediate = grad_output_dtype.transpose(-2, -1) @ x 
        if len(grad_L_wrt_wq_intermediate.shape) > 2: 
            grad_L_wrt_wq_intermediate = grad_L_wrt_wq_intermediate.sum(dim=0)
        grad_L_wrt_wq = grad_L_wrt_wq_intermediate.to(w_q.dtype)
        
        if effective_group_size != -1:
            ste_mask_grouped = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
            ste_mask = ste_mask_grouped.reshape(w_comb.shape)
        else:
            ste_mask = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
        
        grad_L_wrt_wq_masked_by_ste = grad_L_wrt_wq * ste_mask.to(grad_L_wrt_wq.dtype)

        if ctx.needs_input_grad[2]: 
            grad_lora_a = alpha * (lora_b.transpose(0,1) @ grad_L_wrt_wq_masked_by_ste)
        if ctx.needs_input_grad[3]: 
            grad_lora_b = (grad_L_wrt_wq_masked_by_ste @ lora_a.transpose(0,1)) * alpha
        
        if ctx.needs_input_grad[6]: # q_scale
            quantized_w_int_casted = quantized_w_int.to(grad_L_wrt_wq.dtype)
            if effective_group_size != -1:
                num_groups_w = w_comb.numel() // effective_group_size
                grad_L_wrt_wq_grouped = grad_L_wrt_wq.reshape(num_groups_w, effective_group_size)
                grad_q_scale_grouped = (grad_L_wrt_wq_grouped * quantized_w_int_casted).sum(dim=1) 
                grad_q_scale = grad_q_scale_grouped.to(q_scale.dtype)
            else: # Per-tensor
                grad_q_scale_sum = (grad_L_wrt_wq * quantized_w_int_casted).sum()
                # Pastikan grad_q_scale memiliki shape yang sama dengan q_scale input
                if q_scale.numel() == 1 and q_scale.ndim == 1: # Jika q_scale adalah [1]
                    grad_q_scale = grad_q_scale_sum.reshape(1).to(q_scale.dtype)
                elif q_scale.numel() == 1 and q_scale.ndim == 0: # Jika q_scale adalah skalar (jarang terjadi jika Parameter)
                     grad_q_scale = grad_q_scale_sum.to(q_scale.dtype)
                else: # Kasus lain (seharusnya tidak terjadi jika q_scale per-tensor adalah [1])
                    grad_q_scale = grad_q_scale_sum.to(q_scale.dtype) # Default ke skalar jika bingung
        
        print("\n--- Debug Gradients L4QQuantizedLinearFunction.backward ---")
        check_tensor_grad(grad_x, "grad_x (linear)")
        print(f"  DEBUG GRAD: grad_w0 (linear): {grad_w0}") 
        check_tensor_grad(grad_lora_a, "grad_lora_a (linear)")
        check_tensor_grad(grad_lora_b, "grad_lora_b (linear)")
        check_tensor_grad(grad_q_scale, "grad_q_scale (linear)")
        print("--- Akhir Debug Gradients Linear ---")

        return grad_x, grad_w0, grad_lora_a, grad_lora_b, None, None, grad_q_scale, None


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
        self.group_size = group_size 

        self.w0 = Parameter(torch.Tensor(out_features, in_features))
        self.lora_b = Parameter(torch.Tensor(out_features, lora_rank))
        self.lora_a = Parameter(torch.Tensor(lora_rank, in_features))
        
        num_elements = out_features * in_features
        self.effective_group_size_init = self.group_size
        if self.group_size != -1:
            if num_elements == 0:
                 self.effective_group_size_init = -1 
            elif num_elements < self.group_size or num_elements % self.group_size != 0 :
                # print(f"L4QLinear INFO: num_elements ({num_elements}) or divisibility issue with group_size ({self.group_size}). Using per-tensor for q_scale init.")
                self.effective_group_size_init = -1 
        
        if self.effective_group_size_init != -1:
            if num_elements == 0 : # Jika num_elements 0, num_groups akan 0, q_scale akan kosong
                self.q_scale = Parameter(torch.Tensor(0)) # Tensor kosong
            else:
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
            if self.w0.numel() > 0: # Hanya inisialisasi skala jika bobot ada
                w_lora_init = self.alpha * (self.lora_b @ self.lora_a)
                w_comb_init = self.w0 + w_lora_init
                initial_scale = l4q_init_scale(w_comb_init, self.n_bits, self.effective_group_size_init)
                
                if self.q_scale.numel() > 0 : # Hanya copy jika q_scale tidak kosong
                    if self.effective_group_size_init != -1 and initial_scale.numel() == self.q_scale.numel():
                        self.q_scale.data.copy_(initial_scale) 
                    elif initial_scale.numel() == 1: 
                        self.q_scale.data.fill_(initial_scale.item())
                    elif initial_scale.numel() > 0 : # Jika initial_scale adalah tensor tapi tidak cocok (jarang terjadi jika logika benar)
                        print(f"WARNING L4QLinear: Mismatch in q_scale initialization. initial_scale shape: {initial_scale.shape}, self.q_scale shape: {self.q_scale.shape}. Filling with first element.")
                        self.q_scale.data.fill_(initial_scale[0].item())
                    else: # initial_scale kosong (misalnya w_comb_init kosong)
                        self.q_scale.data.fill_(1e-9) # Default kecil
            elif self.q_scale.numel() > 0 : # Jika w0 kosong tapi q_scale ada (misalnya per-tensor)
                 self.q_scale.data.fill_(1e-9)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = L4QQuantizedLinearFunction.apply(x, self.w0, self.lora_a, self.lora_b, 
                                                  self.alpha, self.n_bits, self.q_scale,
                                                  self.group_size) 
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'lora_rank={self.lora_rank}, n_bits={self.n_bits}, alpha={self.alpha}, '
                f'group_size={self.group_size}, bias={self.bias is not None}, '
                f'q_scale_shape={self.q_scale.shape}')

