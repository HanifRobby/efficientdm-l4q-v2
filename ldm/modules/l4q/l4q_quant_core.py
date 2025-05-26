import torch

# Tidak ada impor dari .l4q_linear_layer atau .l4q_conv2d_layer di sini

def get_quantization_bounds(n_bits: int):
    """
    Menghitung batas bawah (Q_N) dan batas atas (Q_P) untuk kuantisasi n-bit.
    """
    q_n = -2**(n_bits - 1)
    q_p = 2**(n_bits - 1) - 1
    return q_n, q_p

def l4q_init_scale(weight_tensor: torch.Tensor, n_bits: int, group_size: int = -1):
    """
    Menginisialisasi parameter skala (s) untuk kuantisasi menggunakan metode L4Q_init.
    """
    q_n, q_p = get_quantization_bounds(n_bits)

    if group_size == -1 or weight_tensor.numel() == 0 or weight_tensor.numel() <= group_size: # Per-tensor atau jika tensor terlalu kecil/kosong
        # Handle tensor kosong
        if weight_tensor.numel() == 0:
            return torch.tensor(1e-9, device=weight_tensor.device, dtype=weight_tensor.dtype)

        w_min = weight_tensor.min().item()
        w_max = weight_tensor.max().item()
        
        scale_n = abs(w_min / q_n) if q_n != 0 and w_min != 0 else float('inf')
        scale_p = abs(w_max / q_p) if q_p != 0 and w_max != 0 else float('inf')

        current_scale = 0.0
        if scale_n == float('inf') and scale_p == float('inf'): 
            current_scale = 1e-9 
        elif scale_n == float('inf'):
            current_scale = scale_p
        elif scale_p == float('inf'):
            current_scale = scale_n
        else:
            current_scale = max(scale_n, scale_p)
        
        if current_scale == 0: # Mencegah skala nol
            current_scale = 1e-9
            
        return torch.tensor(current_scale, device=weight_tensor.device, dtype=weight_tensor.dtype)
    else: # Group-wise quantization
        if weight_tensor.numel() % group_size != 0:
            print(f"PERINGATAN l4q_quant_core.py: Ukuran tensor bobot ({weight_tensor.numel()}) tidak dapat dibagi habis oleh group_size ({group_size}). Menggunakan per-tensor untuk bobot ini.")
            return l4q_init_scale(weight_tensor, n_bits, group_size=-1) # Fallback
        
        num_groups = weight_tensor.numel() // group_size
        weight_groups = weight_tensor.reshape(num_groups, group_size)
        
        scales = torch.zeros(num_groups, device=weight_tensor.device, dtype=weight_tensor.dtype)
        
        for i in range(num_groups):
            w_min_group = weight_groups[i].min().item()
            w_max_group = weight_groups[i].max().item()

            scale_n_group = abs(w_min_group / q_n) if q_n != 0 and w_min_group != 0 else float('inf')
            scale_p_group = abs(w_max_group / q_p) if q_p != 0 and w_max_group != 0 else float('inf')

            current_scale_group = 0.0
            if scale_n_group == float('inf') and scale_p_group == float('inf'):
                current_scale_group = 1e-9 
            elif scale_n_group == float('inf'):
                current_scale_group = scale_p_group
            elif scale_p_group == float('inf'):
                current_scale_group = scale_n_group
            else:
                current_scale_group = max(scale_n_group, scale_p_group)
            
            if current_scale_group == 0: # Mencegah skala nol
                current_scale_group = 1e-9
            scales[i] = current_scale_group
            
        return scales
