import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

# Asumsikan l4q_utils berada di path yang benar atau sesuaikan impornya
# Kita akan menggunakan get_quantization_bounds dan l4q_init_scale dari sana
from .l4q_utils import get_quantization_bounds, l4q_init_scale

class L4QQuantizedConv2dFunction(torch.autograd.Function):
    """
    Fungsi autograd kustom untuk layer Conv2d terkuantisasi L4Q.
    """
    @staticmethod
    def forward(ctx, x, w0, lora_a, lora_b, bias,
                alpha, n_bits, q_scale, quant_group_size,
                stride, padding, dilation, groups):

        # LoRA untuk bobot Conv2d:
        # w0: [out_channels, in_channels // groups, kernel_height, kernel_width]
        # lora_a (La): [rank, (in_channels // groups) * kernel_height * kernel_width]
        # lora_b (Lb): [out_channels, rank]
        # delta_W_flat = Lb @ La
        # delta_W = delta_W_flat.view_as(w0)
        kernel_shape_dims = w0.shape[1:] # (in_channels // groups, KH, KW)
        num_kernel_elements_per_filter = math.prod(kernel_shape_dims)

        delta_w_flat = lora_b @ lora_a # Shape: [out_channels, num_kernel_elements_per_filter]
        delta_w = delta_w_flat.view(w0.shape[0], *kernel_shape_dims) # Shape: [out_channels, C_in/G, KH, KW]
        
        w_comb = w0 + alpha * delta_w

        # Kuantisasi W_comb
        q_n, q_p = get_quantization_bounds(n_bits)
        
        if quant_group_size != -1:
            original_shape = w_comb.shape
            # Reshape untuk group-wise quantization
            # Pastikan numel bisa dibagi habis oleh quant_group_size
            if w_comb.numel() % quant_group_size != 0 and w_comb.numel() > quant_group_size : # Cek jika tidak bisa dibagi habis kecuali jika quant_group_size > numel (artinya per-tensor)
                 # Jika tensor lebih kecil dari group_size, anggap per-tensor
                if w_comb.numel() < quant_group_size:
                    # Fallback ke per-tensor jika tensor lebih kecil dari group_size
                    effective_quant_group_size = -1
                    w_scaled = w_comb / q_scale # q_scale harus skalar di sini
                    quantized_w_int = torch.round(torch.clamp(w_scaled, q_n, q_p))
                    w_q = quantized_w_int * q_scale
                    w_scaled_for_ste = w_scaled
                else:
                    raise ValueError(f"Ukuran tensor bobot ({w_comb.numel()}) harus dapat dibagi habis oleh quant_group_size ({quant_group_size}).")
            else:
                effective_quant_group_size = quant_group_size
                if w_comb.numel() <= quant_group_size: # Jika tensor <= group_size, anggap per-tensor
                    effective_quant_group_size = -1 # Fallback
                
            if effective_quant_group_size != -1:
                num_quant_groups = w_comb.numel() // effective_quant_group_size
                w_comb_grouped = w_comb.reshape(num_quant_groups, effective_quant_group_size)
                
                if q_scale.numel() == num_quant_groups:
                    scale_expanded = q_scale.unsqueeze(1)
                elif q_scale.numel() == 1: # Per-tensor scale broadcast ke semua grup
                    scale_expanded = q_scale
                else:
                    raise ValueError(f"Shape q_scale ({q_scale.shape}) tidak cocok untuk group-wise quantization ({num_quant_groups} groups).")

                w_scaled_grouped = w_comb_grouped / scale_expanded
                quantized_w_int_grouped = torch.round(torch.clamp(w_scaled_grouped, q_n, q_p))
                w_q_grouped = quantized_w_int_grouped * scale_expanded
                w_q = w_q_grouped.reshape(original_shape)
                w_scaled_for_ste = w_scaled_grouped # Akan di-reshape di backward jika perlu
            # Fallback case sudah ditangani di atas
        else: # Per-tensor quantization
            w_scaled = w_comb / q_scale
            quantized_w_int = torch.round(torch.clamp(w_scaled, q_n, q_p))
            w_q = quantized_w_int * q_scale
            w_scaled_for_ste = w_scaled

        output = F.conv2d(x, w_q, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(x, w0, lora_a, lora_b, bias, q_scale, w_q, quantized_w_int, w_comb, delta_w)
        ctx.alpha = alpha
        ctx.n_bits = n_bits
        ctx.quant_group_size = quant_group_size
        ctx.q_n, ctx.q_p = q_n, q_p
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.kernel_shape_dims = kernel_shape_dims
        ctx.w_scaled_for_ste = w_scaled_for_ste # Disimpan dalam bentuk grouped atau per-tensor

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w0, lora_a, lora_b, bias, q_scale, w_q, quantized_w_int, w_comb, delta_w = ctx.saved_tensors
        alpha = ctx.alpha
        n_bits = ctx.n_bits
        quant_group_size = ctx.quant_group_size
        q_n, q_p = ctx.q_n, ctx.q_p
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        kernel_shape_dims = ctx.kernel_shape_dims
        w_scaled_for_ste = ctx.w_scaled_for_ste


        grad_x = grad_w0 = grad_lora_a = grad_lora_b = grad_bias = None
        grad_alpha = grad_n_bits = grad_q_scale = grad_quant_group_size_param = None
        grad_stride = grad_padding = grad_dilation = grad_groups = None


        # 1. Gradien untuk input x
        if ctx.needs_input_grad[0]:
            grad_x = F.grad.conv2d_input(x.shape, w_q, grad_output, stride, padding, dilation, groups)

        # 2. Gradien untuk W_q (grad_L_wrt_wq)
        # Ini adalah gradien loss terhadap kernel terkuantisasi W_q
        grad_L_wrt_wq = F.grad.conv2d_weight(x, w_q.shape, grad_output, stride, padding, dilation, groups)
        
        # --- STE Mask untuk clamping ---
        if quant_group_size != -1 and w_comb.numel() > quant_group_size: # Hanya jika group-wise benar-benar terjadi
            # w_scaled_for_ste disimpan sebagai grouped
            ste_mask_grouped = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
            ste_mask = ste_mask_grouped.reshape(w_comb.shape)
        else: # Per-tensor
            ste_mask = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)

        grad_L_wrt_wq_masked_by_ste = grad_L_wrt_wq * ste_mask
        
        # Gradien dL/dW_comb_effective = grad_L_wrt_wq_masked_by_ste (dengan asumsi dWq/dWcomb_eff = 1 via STE)
        # Gradien dL/d_delta_w = alpha * grad_L_wrt_wq_masked_by_ste
        grad_L_wrt_delta_w = alpha * grad_L_wrt_wq_masked_by_ste
        
        # Reshape grad_L_wrt_delta_w menjadi flat untuk menghitung gradien LoRA
        grad_L_wrt_delta_w_flat = grad_L_wrt_delta_w.reshape(w0.shape[0], -1) # [out_channels, num_kernel_elements_per_filter]

        # 3. Gradien untuk parameter LoRA A dan B
        if ctx.needs_input_grad[2]: # lora_a
            # dL/dLa = Lb^T @ dL/d_delta_w_flat
            grad_lora_a = lora_b.transpose(0, 1) @ grad_L_wrt_delta_w_flat
        
        if ctx.needs_input_grad[3]: # lora_b
            # dL/dLb = dL/d_delta_w_flat @ La^T
            grad_lora_b = grad_L_wrt_delta_w_flat @ lora_a.transpose(0, 1)

        # 4. Gradien untuk parameter kuantisasi (q_scale)
        # dL/ds = sum(dL/dWq_i * quantized_w_int_i)
        if ctx.needs_input_grad[7]: # q_scale
            if quant_group_size != -1 and w_comb.numel() > quant_group_size:
                num_quant_groups = w_comb.numel() // quant_group_size
                
                # Pastikan quantized_w_int memiliki shape yang sesuai (sudah grouped atau perlu di-reshape)
                # Jika quantized_w_int disimpan sbg tensor asli, reshape sekarang
                if quantized_w_int.shape != (num_quant_groups, quant_group_size):
                    quantized_w_int_grouped = quantized_w_int.reshape(num_quant_groups, quant_group_size)
                else:
                    quantized_w_int_grouped = quantized_w_int # Sudah grouped dari forward (jika w_scaled_for_ste adalah grouped)

                grad_L_wrt_wq_grouped = grad_L_wrt_wq.reshape(num_quant_groups, quant_group_size)
                
                grad_q_scale_grouped = (grad_L_wrt_wq_grouped * quantized_w_int_grouped).sum(dim=1)
                grad_q_scale = grad_q_scale_grouped # Shape [num_quant_groups]
            else: # Per-tensor
                grad_q_scale = (grad_L_wrt_wq * quantized_w_int).sum()
        
        # 5. Gradien untuk bias (jika ada dan trainable)
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=[0, 2, 3]) # Sum over batch and spatial dimensions

        return (grad_x, grad_w0, grad_lora_a, grad_lora_b, grad_bias,
                grad_alpha, grad_n_bits, grad_q_scale, grad_quant_group_size_param,
                grad_stride, grad_padding, grad_dilation, grad_groups)


class L4QQuantizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 lora_rank: int, n_bits: int, alpha: float = 1.0,
                 quant_group_size: int = -1, # -1 untuk per-tensor
                 stride = 1, padding = 0, dilation = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        self.lora_rank = lora_rank
        self.n_bits = n_bits
        self.alpha = alpha
        self.quant_group_size = quant_group_size
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Bobot asli W0 (dibekukan)
        self.w0 = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        
        # Parameter LoRA B dan A
        # lora_a (La): [rank, (in_channels // groups) * KH * KW]
        # lora_b (Lb): [out_channels, rank]
        num_kernel_elements_per_filter = (in_channels // groups) * kernel_size[0] * kernel_size[1]
        self.lora_a = Parameter(torch.Tensor(lora_rank, num_kernel_elements_per_filter))
        self.lora_b = Parameter(torch.Tensor(out_channels, lora_rank))
        
        # Parameter Kuantisasi (skala 's')
        num_weight_elements = self.w0.numel()
        if quant_group_size != -1:
            if num_weight_elements < quant_group_size : # Fallback jika tensor lebih kecil
                effective_quant_group_size = num_weight_elements 
            else:
                effective_quant_group_size = quant_group_size

            if num_weight_elements % effective_quant_group_size != 0 :
                 # Jika tensor lebih kecil dari group_size, anggap per-tensor
                if num_weight_elements < effective_quant_group_size:
                    self.q_scale = Parameter(torch.Tensor(1)) # Per-tensor
                    self.quant_group_size = -1 # Update karena fallback
                else:
                    raise ValueError(f"Jumlah elemen bobot ({num_weight_elements}) harus habis dibagi quant_group_size ({effective_quant_group_size}).")
            else:
                num_quant_groups = num_weight_elements // effective_quant_group_size
                self.q_scale = Parameter(torch.Tensor(num_quant_groups))
        else: # Per-tensor
            self.q_scale = Parameter(torch.Tensor(1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.w0.requires_grad = False # Bekukan W0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w0)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        with torch.no_grad():
            kernel_shape_dims = self.w0.shape[1:]
            delta_w_flat_init = self.lora_b @ self.lora_a
            delta_w_init = delta_w_flat_init.view(self.out_channels, *kernel_shape_dims)
            w_comb_init = self.w0 + self.alpha * delta_w_init
            
            initial_scale = l4q_init_scale(w_comb_init, self.n_bits, self.quant_group_size) # Gunakan self.quant_group_size yang mungkin sudah diupdate
            
            if self.quant_group_size != -1:
                 self.q_scale.data.copy_(initial_scale.squeeze())
            else:
                 self.q_scale.data.copy_(initial_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return L4QQuantizedConv2dFunction.apply(
            x, self.w0, self.lora_a, self.lora_b, self.bias,
            self.alpha, self.n_bits, self.q_scale, self.quant_group_size,
            self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += (f', lora_rank={self.lora_rank}, n_bits={self.n_bits}, alpha={self.alpha}, '
              f'quant_group_size={self.quant_group_size}')
        return s.format(**self.__dict__)