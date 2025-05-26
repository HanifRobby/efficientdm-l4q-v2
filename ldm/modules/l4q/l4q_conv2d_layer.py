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

class L4QQuantizedConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, lora_a, lora_b, bias,
                alpha, n_bits, q_scale, quant_group_size,
                stride, padding, dilation, groups):
        # ... (Kode forward Anda yang sudah ada, pastikan effective_qgs_fwd dihitung dengan benar) ...
        kernel_shape_dims = w0.shape[1:] 
        delta_w_flat = lora_b @ lora_a 
        delta_w = delta_w_flat.view(w0.shape[0], *kernel_shape_dims) 
        w_comb = w0 + alpha * delta_w
        q_n, q_p = get_quantization_bounds(n_bits)
        
        effective_qgs_fwd = quant_group_size
        if quant_group_size != -1:
            if w_comb.numel() == 0 or w_comb.numel() < quant_group_size or w_comb.numel() % quant_group_size != 0:
                effective_qgs_fwd = -1 
        
        w_scaled_grouped = None
        w_scaled = None

        if effective_qgs_fwd != -1:
            original_shape = w_comb.shape
            num_quant_groups = w_comb.numel() // effective_qgs_fwd
            w_comb_grouped = w_comb.reshape(num_quant_groups, effective_qgs_fwd)
            
            current_q_scale = q_scale
            if q_scale.numel() == num_quant_groups:
                scale_expanded = current_q_scale.unsqueeze(1)
            elif q_scale.numel() == 1: 
                scale_expanded = current_q_scale
            else:
                scale_expanded = current_q_scale 
                effective_qgs_fwd = -1
            
            if effective_qgs_fwd != -1:
                 w_scaled_grouped = w_comb_grouped / (scale_expanded + 1e-9)
        
        if effective_qgs_fwd == -1: 
            w_scaled = w_comb / (q_scale + 1e-9) if q_scale.numel() > 1 else w_comb / (q_scale.item() + 1e-9)

        if effective_qgs_fwd != -1:
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
            
        output = F.conv2d(x, w_q, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(x, w0, lora_a, lora_b, bias, q_scale, w_q, quantized_w_int_for_backward, w_comb, delta_w)
        ctx.alpha = alpha
        ctx.q_n, ctx.q_p = q_n, q_p
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.w_scaled_for_ste = w_scaled_for_ste
        ctx.effective_quant_group_size_used_in_fwd = effective_qgs_fwd
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w0, lora_a, lora_b, bias, q_scale, w_q, quantized_w_int, w_comb, delta_w = ctx.saved_tensors
        alpha = ctx.alpha
        effective_qgs = ctx.effective_quant_group_size_used_in_fwd
        q_n, q_p = ctx.q_n, ctx.q_p
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        w_scaled_for_ste = ctx.w_scaled_for_ste

        grad_x = grad_w0 = grad_lora_a = grad_lora_b = grad_bias = grad_q_scale = None
        
        grad_output_dtype = grad_output.to(x.dtype)

        if ctx.needs_input_grad[0]:
            grad_x = F.grad.conv2d_input(x.shape, w_q.to(x.dtype), grad_output_dtype, stride, padding, dilation, groups)
        
        grad_L_wrt_wq_intermediate = F.grad.conv2d_weight(x, w_q.shape, grad_output_dtype, stride, padding, dilation, groups)
        grad_L_wrt_wq = grad_L_wrt_wq_intermediate.to(w_q.dtype)
        
        if effective_qgs != -1:
            ste_mask_grouped = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
            ste_mask = ste_mask_grouped.reshape(w_comb.shape)
        else:
            ste_mask = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
        
        grad_L_wrt_wq_masked_by_ste = grad_L_wrt_wq * ste_mask.to(grad_L_wrt_wq.dtype)
        grad_L_wrt_delta_w = alpha * grad_L_wrt_wq_masked_by_ste
        grad_L_wrt_delta_w_flat = grad_L_wrt_delta_w.reshape(w0.shape[0], -1)

        if ctx.needs_input_grad[2]: 
            grad_lora_a = lora_b.transpose(0, 1) @ grad_L_wrt_delta_w_flat
        if ctx.needs_input_grad[3]: 
            grad_lora_b = grad_L_wrt_delta_w_flat @ lora_a.transpose(0, 1)
        
        if ctx.needs_input_grad[7]: # q_scale
            quantized_w_int_casted = quantized_w_int.to(grad_L_wrt_wq.dtype)
            if effective_qgs != -1:
                num_quant_groups = w_comb.numel() // effective_qgs
                grad_L_wrt_wq_grouped = grad_L_wrt_wq.reshape(num_quant_groups, effective_qgs)
                grad_q_scale_grouped = (grad_L_wrt_wq_grouped * quantized_w_int_casted).sum(dim=1)
                grad_q_scale = grad_q_scale_grouped.to(q_scale.dtype)
            else: # Per-tensor
                grad_q_scale_sum = (grad_L_wrt_wq * quantized_w_int_casted).sum()
                # Pastikan grad_q_scale memiliki shape yang sama dengan q_scale input
                if q_scale.numel() == 1 and q_scale.ndim == 1: # Jika q_scale adalah [1]
                    grad_q_scale = grad_q_scale_sum.reshape(1).to(q_scale.dtype)
                elif q_scale.numel() == 1 and q_scale.ndim == 0: # Jika q_scale adalah skalar
                     grad_q_scale = grad_q_scale_sum.to(q_scale.dtype)
                else:
                    grad_q_scale = grad_q_scale_sum.to(q_scale.dtype)
        
        if bias is not None and ctx.needs_input_grad[4]: # bias
            grad_bias_intermediate = grad_output_dtype.sum(dim=[0, 2, 3])
            grad_bias = grad_bias_intermediate.to(bias.dtype)
            if len(grad_bias.shape) == 0 and bias.numel() == 1: pass 
            elif len(grad_bias.shape) > 0 and bias.numel() == 1 and grad_bias.numel() == 1: grad_bias = grad_bias.squeeze()
        
        print("\n--- Debug Gradients L4QQuantizedConv2dFunction.backward ---")
        check_tensor_grad(grad_x, "grad_x (conv2d)")
        print(f"  DEBUG GRAD: grad_w0 (conv2d): {grad_w0}")
        check_tensor_grad(grad_lora_a, "grad_lora_a (conv2d)")
        check_tensor_grad(grad_lora_b, "grad_lora_b (conv2d)")
        check_tensor_grad(grad_bias, "grad_bias (conv2d)")
        check_tensor_grad(grad_q_scale, "grad_q_scale (conv2d)")
        print("--- Akhir Debug Gradients Conv2d ---")

        return (grad_x, grad_w0, grad_lora_a, grad_lora_b, grad_bias,
                None, None, grad_q_scale, None, 
                None, None, None, None)


class L4QQuantizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 lora_rank: int, n_bits: int, alpha: float = 1.0,
                 quant_group_size: int = -1, 
                 stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        if isinstance(kernel_size, int): self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        self.lora_rank = lora_rank; self.n_bits = n_bits; self.alpha = alpha
        self.quant_group_size = quant_group_size
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.groups = groups

        self.w0 = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        num_kernel_elements_per_filter = (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        self.lora_a = Parameter(torch.Tensor(lora_rank, num_kernel_elements_per_filter))
        self.lora_b = Parameter(torch.Tensor(out_channels, lora_rank))
        
        num_weight_elements = self.w0.numel()
        self.effective_group_size_init = self.quant_group_size
        if self.quant_group_size != -1:
            if num_weight_elements == 0:
                self.effective_group_size_init = -1
            elif num_weight_elements < self.quant_group_size or num_weight_elements % self.quant_group_size != 0:
                # print(f"L4QConv2d INFO: num_elements ({num_weight_elements}) or divisibility issue with q_gs ({self.quant_group_size}). Using per-tensor for q_scale init.")
                self.effective_group_size_init = -1
        
        if self.effective_group_size_init != -1:
            if num_weight_elements == 0:
                self.q_scale = Parameter(torch.Tensor(0))
            else:
                num_quant_groups = num_weight_elements // self.effective_group_size_init
                self.q_scale = Parameter(torch.Tensor(num_quant_groups))
        else:
            self.q_scale = Parameter(torch.Tensor(1))

        if bias: self.bias = Parameter(torch.Tensor(out_channels))
        else: self.register_parameter('bias', None)
        self.reset_parameters()
        self.w0.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w0)
            if fan_in != 0: bound = 1 / math.sqrt(fan_in); nn.init.uniform_(self.bias, -bound, bound)
            else: nn.init.zeros_(self.bias)

        with torch.no_grad():
            if self.w0.numel() > 0:
                kernel_shape_dims = self.w0.shape[1:]
                delta_w_flat_init = self.lora_b @ self.lora_a
                delta_w_init = delta_w_flat_init.view(self.out_channels, *kernel_shape_dims)
                w_comb_init = self.w0 + self.alpha * delta_w_init
                initial_scale = l4q_init_scale(w_comb_init, self.n_bits, self.effective_group_size_init)
                
                if self.q_scale.numel() > 0:
                    if self.effective_group_size_init != -1 and initial_scale.numel() == self.q_scale.numel():
                        self.q_scale.data.copy_(initial_scale)
                    elif initial_scale.numel() == 1:
                        self.q_scale.data.fill_(initial_scale.item())
                    elif initial_scale.numel() > 0:
                        print(f"WARNING L4QConv2d: Mismatch in q_scale init. initial_scale: {initial_scale.shape}, self.q_scale: {self.q_scale.shape}. Filling with first.")
                        self.q_scale.data.fill_(initial_scale[0].item())
                    else:
                         self.q_scale.data.fill_(1e-9)
            elif self.q_scale.numel() > 0:
                 self.q_scale.data.fill_(1e-9)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return L4QQuantizedConv2dFunction.apply(
            x, self.w0, self.lora_a, self.lora_b, self.bias,
            self.alpha, self.n_bits, self.q_scale, self.quant_group_size,
            self.stride, self.padding, self.dilation, self.groups
        )
    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        # Safely format padding and dilation
        padding_str = str(self.padding)
        dilation_str = str(self.dilation)
        if (isinstance(self.padding, tuple) and any(p != 0 for p in self.padding)) or \
           (isinstance(self.padding, int) and self.padding != 0):
            s += f', padding={padding_str}'
        if (isinstance(self.dilation, tuple) and any(d != 1 for d in self.dilation)) or \
           (isinstance(self.dilation, int) and self.dilation != 1):
            s += f', dilation={dilation_str}'

        if self.groups != 1: s += ', groups={groups}'
        if self.bias is None: s += ', bias=False'
        s += (f', lora_rank={self.lora_rank}, n_bits={self.n_bits}, alpha={self.alpha}, '
              f'quant_group_size={self.quant_group_size}, q_scale_shape={self.q_scale.shape}')
        # Menggunakan __dict__ bisa berbahaya jika ada atribut dinamis atau non-string keys
        # Lebih aman untuk memformat secara eksplisit atau dengan hati-hati.
        # Untuk sekarang, kita akan format parameter yang diketahui.
        # return s.format(**self.__dict__) # Hindari ini jika memungkinkan
        
        # Format manual untuk menghindari error dengan __dict__
        return s.format(in_channels=self.in_channels, out_channels=self.out_channels, 
                        kernel_size=self.kernel_size, stride=self.stride, 
                        padding=padding_str, dilation=dilation_str, groups=self.groups)

