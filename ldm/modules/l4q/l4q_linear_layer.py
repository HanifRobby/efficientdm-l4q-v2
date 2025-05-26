import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Asumsikan l4q_utils berada di path yang benar atau sesuaikan impornya
from .l4q_utils import l4q_init_scale, get_quantization_bounds

class L4QQuantizedLinearFunction(torch.autograd.Function):
    """
    Fungsi autograd kustom untuk layer linear terkuantisasi L4Q.
    Mengimplementasikan forward pass dengan kuantisasi dan backward pass kustom
    untuk parameter LoRA dan parameter kuantisasi.
    """
    @staticmethod
    def forward(ctx, x, w0, lora_a, lora_b, alpha, n_bits, q_scale, group_size):
        # Simpan tensor dan parameter yang dibutuhkan untuk backward pass
        # q_scale akan dihitung ulang dari w_comb jika diinisialisasi saat itu
        # atau disimpan jika sudah fixed/learnable
        
        w_lora = alpha * (lora_b @ lora_a) # Bentuk W_LoRA = alpha * B @ A [cite: 80]
        w_comb = w0 + w_lora # Gabungkan bobot asli dengan bobot LoRA [cite: 80]

        # Kuantisasi W_comb
        # Untuk group-wise, q_scale akan memiliki shape yang berbeda dan perlu di-handle
        if group_size != -1:
            original_shape = w_comb.shape
            num_groups_w = w_comb.numel() // group_size
            w_comb_grouped = w_comb.view(num_groups_w, group_size)
            
            # Pastikan q_scale di-broadcast dengan benar untuk setiap grup
            if q_scale.numel() == num_groups_w:
                scale_expanded = q_scale.unsqueeze(1) # Shape [num_groups, 1]
            else: # per-tensor scale dipakai untuk semua grup jika tidak cocok
                scale_expanded = q_scale 
            
            w_scaled_grouped = w_comb_grouped / scale_expanded # Pembagian elemen-wise
        else: # Per-tensor quantization
            w_scaled = w_comb / q_scale
        
        q_n, q_p = get_quantization_bounds(n_bits)
        
        if group_size != -1:
            quantized_w_int_grouped = torch.round(torch.clamp(w_scaled_grouped, q_n, q_p))
            w_q_grouped = quantized_w_int_grouped * scale_expanded
            w_q = w_q_grouped.view(original_shape) # Kembalikan ke shape asli
        else:
            quantized_w_int = torch.round(torch.clamp(w_scaled, q_n, q_p))
            w_q = quantized_w_int * q_scale # Dequantize untuk forward pass [cite: 48, 9]

        # Output dari layer linear
        output = F.linear(x, w_q) # Y = W_q * X [cite: 10]
        
        # Simpan variabel yang dibutuhkan untuk backward pass
        # STE mask akan bergantung pada w_scaled (atau w_scaled_grouped)
        ctx.save_for_backward(x, w0, lora_a, lora_b, q_scale, w_q, quantized_w_int, w_comb)
        ctx.alpha = alpha
        ctx.n_bits = n_bits
        ctx.group_size = group_size
        ctx.q_n = q_n
        ctx.q_p = q_p
        
        if group_size != -1:
            ctx.w_scaled_for_ste = w_scaled_grouped.view_as(quantized_w_int_grouped)
        else:
            ctx.w_scaled_for_ste = w_scaled

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Ambil variabel yang disimpan dari forward pass
        x, w0, lora_a, lora_b, q_scale, w_q, quantized_w_int, w_comb = ctx.saved_tensors
        alpha = ctx.alpha
        n_bits = ctx.n_bits
        group_size = ctx.group_size
        q_n = ctx.q_n
        q_p = ctx.q_p
        w_scaled_for_ste = ctx.w_scaled_for_ste # Ini adalah (W_comb / q_scale) sebelum clamp dan round

        grad_x = grad_w0 = grad_lora_a = grad_lora_b = grad_q_scale = None
        grad_alpha = grad_n_bits = grad_group_size_param = None # Tidak ada gradien untuk ini

        # 1. Hitung gradien untuk input x (grad_x)
        if ctx.needs_input_grad[0]: # Index 0 untuk x
            grad_x = F.linear(grad_output, w_q.transpose(0, 1))

        # 2. Hitung gradien terhadap W_q (grad_L_wrt_wq)
        # Ini adalah gradien loss terhadap output dari fungsi kuantisasi W_q
        # Sumber: [cite: 83, 142] (dinyatakan sebagai dL/dWq = dL/dY * X^T)
        grad_L_wrt_wq = grad_output.transpose(-2, -1) @ x 
        if len(grad_L_wrt_wq.shape) > 2: # Handle batch dimension in grad_output and x
            grad_L_wrt_wq = grad_L_wrt_wq.sum(dim=0)


        # --- STE Mask untuk clamping ---
        # STE_mask bernilai 1 jika bobot berada dalam rentang kuantisasi, 0 jika di luar (diklem)
        # Ini akan digunakan untuk mengkondisikan gradien yang melalui operasi non-linear clamp
        # Paper L4Q (Persamaan 13, 14) menyiratkan kondisi ini [cite: 90]
        if group_size != -1:
            original_shape = w_comb.shape
            ste_mask_grouped = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
            ste_mask = ste_mask_grouped.view(original_shape)
        else:
            ste_mask = (w_scaled_for_ste >= q_n) & (w_scaled_for_ste <= q_p)
        
        # Gradien yang "lolos" dari STE pada W_comb
        # dL/dW_comb = dL/dW_q * dW_q/dW_comb
        # dW_q/dW_comb disimplifikasi oleh STE menjadi 1/q_scale jika dalam range, atau q_scale jika dL/dW_q dikalikan q_scale
        # Berdasarkan LSQ+, gradien terhadap float input dari STE adalah gradien output STE itu sendiri.
        # Jadi, dL/dW_comb_pre_clamp_round = grad_L_wrt_wq * (1.0 / q_scale jika w_q = round(w_comb/s)*s)
        # Atau lebih langsung, gradien yang melewati operasi kuantisasi (round, clamp) kembali ke W_comb
        # adalah grad_L_wrt_wq yang dikondisikan oleh STE mask dan diskalakan dengan 1/q_scale.
        # L4Q menyederhanakan ini dengan menggunakan dL/dWq langsung dalam perhitungan gradien LoRA,
        # dengan STE mask diterapkan pada turunan Wq thd A dan B[cite: 87, 90].
        # Kita akan mengikuti pendekatan L4Q: STE mask diterapkan pada dW_q/dA dan dW_q/dB.
        
        # Gradien "efektif" dari loss terhadap W_q yang memperhatikan STE untuk clamping.
        # grad_L_wrt_wq_ste_effective = grad_L_wrt_wq # Tidak perlu dikalikan ste_mask di sini, tapi di turunan LoRA

        # 3. Hitung gradien untuk parameter LoRA A dan B
        # Sumber: [cite: 85] (dL/dA = dL/dWq * dWq/dA), [cite: 90] (Persamaan 13 & 14 untuk dWq/dA dan dWq/dB)
        # dWq/dA = alpha * B^T (jika dalam range STE, 0 jika di luar)
        # dWq/dB = alpha * A^T (jika dalam range STE, 0 jika di luar)
        # Perhatikan: Paper L4Q Eq 13 & 14 tidak menyertakan 's' (q_scale) secara eksplisit pada dWq/dA
        # Ini mungkin karena dL/dWq sudah mencakup efek dari 's' atau 's' dianggap 1 dalam derivasi itu.
        # Namun, jika Wq = s * W_int, maka dWq/dW_comb = s * STE_round * (1/s) * STE_clamp = STE_round * STE_clamp
        # Mari kita ikuti struktur Persamaan 13 & 14 secara langsung.

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]: # lora_a or lora_b
            # dL/dW_comb_effective = grad_L_wrt_wq * ste_mask # Gradien loss thd W_comb, hanya jika tidak di-clamp
                                                            # Ini adalah interpretasi dari g_Wq_ste di pikiran saya sebelumnya
            # Jika kita menganggap dL/dWq adalah gradien efektif setelah dekuantisasi:
            # dL/dA = alpha * B^T @ (dL/dW_q * ste_mask)
            # dL/dB = (dL/dW_q * ste_mask) @ alpha * A^T
            
            # Transpose grad_L_wrt_wq agar sesuai untuk perkalian matriks dengan B^T dan A^T
            # Jika grad_L_wrt_wq adalah (out_features, in_features)
            # lora_b: (out_features, r), lora_a: (r, in_features)
            
            # Gradien dikondisikan oleh STE mask
            grad_L_wrt_wq_masked_by_ste = grad_L_wrt_wq * ste_mask

            if ctx.needs_input_grad[2]: # grad_lora_a (r, in_features)
                # dL/dA = alpha * B^T @ grad_L_wrt_wq_masked_by_ste
                grad_lora_a = alpha * (lora_b.transpose(0,1) @ grad_L_wrt_wq_masked_by_ste)
                
            if ctx.needs_input_grad[3]: # grad_lora_b (out_features, r)
                # dL/dB = grad_L_wrt_wq_masked_by_ste @ alpha * A^T
                grad_lora_b = (grad_L_wrt_wq_masked_by_ste @ lora_a.transpose(0,1)) * alpha
        
        # 4. Hitung gradien untuk parameter kuantisasi (q_scale)
        # Sumber: [cite: 52, 53] (dL/ds = dL/dWq * dWq/ds)
        # dWq/ds dari L4Q Persamaan 5: -w_scaled_clamped + quantized_w_int, jika QN <= w_scaled <= QP
        # Atau Persamaan 19: dWq/ds = (deriv_r_wrt_w * (- (W_comb-b)/s^2)) + quantized_w_int
        # Mari gunakan versi yang lebih sederhana dWq/ds = quantized_w_int jika STE(round) dianggap identity untuk gradien skala.
        # LSQ+ menggunakan: grad_s = (grad_output_quant_layer * (quantized_value - grad_input_quant_layer_float) / s).clamp(min_val, max_val)
        # Atau, jika W_q = s * W_int, maka dW_q/ds = W_int.
        
        if ctx.needs_input_grad[5]: # q_scale
            # dL/ds = sum(dL/dWq_i * dWq_i/ds) = sum(dL/dWq_i * quantized_w_int_i)
            # Handle group-wise vs per-tensor scale
            if group_size != -1:
                original_shape = w_comb.shape
                num_groups_w = w_comb.numel() // group_size
                
                grad_L_wrt_wq_grouped = grad_L_wrt_wq.view(num_groups_w, group_size)
                quantized_w_int_grouped = quantized_w_int.view(num_groups_w, group_size) # Seharusnya sudah di shape ini dari forward
                
                # dWq/ds untuk setiap grup adalah quantized_w_int dari grup itu
                # grad_q_scale akan menjadi [num_groups]
                grad_q_scale_grouped = (grad_L_wrt_wq_grouped * quantized_w_int_grouped).sum(dim=1)
                grad_q_scale = grad_q_scale_grouped
            else: # Per-tensor
                grad_q_scale = (grad_L_wrt_wq * quantized_w_int).sum()

        # w0 tidak trainable, jadi grad_w0 = None
        return grad_x, grad_w0, grad_lora_a, grad_lora_b, grad_alpha, grad_n_bits, grad_q_scale, grad_group_size_param


class L4QQuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 lora_rank: int, n_bits: int, alpha: float = 1.0, 
                 group_size: int = -1, # -1 untuk per-tensor, >0 untuk group-wise
                 bias: bool = True): # Bias belum diimplementasikan dengan kuantisasi L4Q
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.n_bits = n_bits
        self.alpha = alpha
        self.group_size = group_size

        # Bobot asli W0 (dibekukan)
        self.w0 = Parameter(torch.Tensor(out_features, in_features))
        
        # Parameter LoRA B dan A
        # L4Q: B (o x r), A (r x i) [cite: 38]
        self.lora_b = Parameter(torch.Tensor(out_features, lora_rank)) # Matriks B
        self.lora_a = Parameter(torch.Tensor(lora_rank, in_features))  # Matriks A
        
        # Parameter Kuantisasi (skala 's')
        # Akan diinisialisasi menggunakan L4Q_init
        # Jika group_size != -1, q_scale akan menjadi tensor [num_groups]
        # Jika group_size == -1, q_scale akan menjadi skalar
        num_elements = out_features * in_features
        if group_size != -1:
            if num_elements % group_size != 0:
                raise ValueError("Jumlah elemen bobot harus habis dibagi group_size.")
            num_groups = num_elements // group_size
            self.q_scale = Parameter(torch.Tensor(num_groups))
        else:
            self.q_scale = Parameter(torch.Tensor(1)) # Skalar

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            # Catatan: Kuantisasi bias belum di-cover oleh L4Q secara eksplisit di paper utama.
            # Biasanya bias dikuantisasi dengan bit-width lebih tinggi atau dibiarkan float.
            # Untuk saat ini, bias akan tetap float.
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.w0.requires_grad = False # Bekukan W0 [cite: 40]

    def reset_parameters(self):
        # Inisialisasi standar untuk W0, LoRA A, B, dan bias
        nn.init.kaiming_uniform_(self.w0, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5) # Atau inisialisasi nol untuk lora_b seperti di LoRA asli
        nn.init.zeros_(self.lora_b)
        
        if self.bias is not None:
            bound = 1 / (self.in_features**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Inisialisasi skala kuantisasi menggunakan L4Q_init
        # L4Q_init sebaiknya dipanggil setelah W0, A, B diinisialisasi untuk mendapatkan W_comb awal
        with torch.no_grad():
            w_lora_init = self.alpha * (self.lora_b @ self.lora_a)
            w_comb_init = self.w0 + w_lora_init
            
            # Inisialisasi q_scale
            # Jika group_size != -1, l4q_init_scale akan mengembalikan tensor [num_groups]
            # yang perlu di-reshape menjadi [num_groups, 1] agar sesuai dengan self.q_scale jika 
            # self.q_scale didefinisikan sebagai [num_groups] (bukan [num_groups, 1]).
            # Mari kita pastikan shape-nya konsisten.
            
            initial_scale = l4q_init_scale(w_comb_init, self.n_bits, self.group_size)
            if self.group_size != -1:
                 # l4q_init_scale mengembalikan [num_groups, 1] atau [num_groups]
                 # Parameter q_scale didefinisikan sebagai [num_groups]
                 self.q_scale.data.copy_(initial_scale.squeeze()) # squeeze jika [N,1] jadi [N]
            else: # skalar
                 self.q_scale.data.copy_(initial_scale)


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
                f'group_size={self.group_size}, bias={self.bias is not None}')