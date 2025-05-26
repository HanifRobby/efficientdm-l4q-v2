import torch
import torch.nn as nn # Diperlukan untuk fallback jika L4QQuantized* tidak ditemukan

# Impor layer L4Q dasar Anda
try:
    from .l4q_linear_layer import L4QQuantizedLinear
    from .l4q_conv2d_layer import L4QQuantizedConv2d
except ImportError:
    print("PERINGATAN KRITIKAL di l4q_utils.py: Gagal mengimpor L4QQuantizedLinear/Conv2d. Fungsi helper L4Q akan fallback ke nn.Linear/Conv2d.")
    # Fallback jika layer L4Q dasar tidak ditemukan (ini seharusnya tidak terjadi jika file ada)
    L4QQuantizedLinear = nn.Linear 
    L4QQuantizedConv2d = nn.Conv2d


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

    if group_size == -1 or weight_tensor.numel() <= group_size: # Per-tensor atau jika tensor terlalu kecil untuk group
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
        
        if current_scale == 0:
            current_scale = 1e-9
            
        return torch.tensor(current_scale, device=weight_tensor.device, dtype=weight_tensor.dtype)
    else: # Group-wise quantization
        if weight_tensor.numel() % group_size != 0:
            raise ValueError(f"Ukuran tensor bobot ({weight_tensor.numel()}) harus dapat dibagi habis oleh group_size ({group_size}).")
        
        num_groups = weight_tensor.numel() // group_size
        weight_groups = weight_tensor.reshape(num_groups, group_size) # Menggunakan reshape
        
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
            
            if current_scale_group == 0:
                current_scale_group = 1e-9
            scales[i] = current_scale_group
            
        # Untuk group-wise, q_scale di L4QQuantizedLinear/Conv2d biasanya diharapkan [num_groups]
        # atau [num_groups, 1] jika broadcasting diurus di sana.
        # l4q_init_scale di paper L4Q tidak secara eksplisit menyebut shape output untuk group-wise.
        # Mari kita kembalikan sebagai [num_groups] untuk konsistensi dengan Parameter.
        return scales # Shape [num_groups]

# --- Fungsi Helper L4Q (DIPINDAHKAN KE SINI) ---
def make_l4q_linear(in_features, out_features, bias=True, 
                    lora_rank=4, n_bits=4, alpha=1.0, group_size=-1,
                    l4q_enabled_passed=True # Flag tambahan untuk kontrol eksternal jika diperlukan
                    ):
    """Helper function to create L4QQuantizedLinear layer or fallback to nn.Linear."""
    if l4q_enabled_passed and L4QQuantizedLinear != nn.Linear : # Cek juga fallback L4QQuantizedLinear
        # print(f"DEBUG: Membuat L4QQuantizedLinear ({in_features}, {out_features})")
        return L4QQuantizedLinear(
            in_features, out_features, bias=bias,
            lora_rank=lora_rank, n_bits=n_bits, alpha=alpha, 
            group_size=group_size
        )
    else:
        # print(f"DEBUG: Fallback ke nn.Linear ({in_features}, {out_features})")
        return nn.Linear(in_features, out_features, bias=bias)

def make_l4q_conv2d(dims, # Tambahkan dims untuk kompatibilitas dengan conv_nd OpenAI
                    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                    lora_rank=4, n_bits=4, alpha=1.0, quant_group_size=-1,
                    l4q_enabled_passed=True # Flag tambahan
                    ):
    """Helper function to create L4QQuantizedConv2d layer or fallback to nn.ConvXd."""
    # Parameter 'dims' dari conv_nd OpenAI tidak secara langsung dipakai oleh L4QQuantizedConv2d
    # karena L4QQuantizedConv2d kita asumsikan 2D.
    # Jika Anda perlu mendukung conv1d/3d L4Q, Anda perlu implementasi terpisah.
    if dims != 2 and l4q_enabled_passed and L4QQuantizedConv2d != nn.Conv2d:
        print(f"PERINGATAN: make_l4q_conv2d dipanggil dengan dims={dims} tapi hanya mendukung dims=2 untuk L4Q. Fallback ke nn.Conv{dims}d.")
        l4q_enabled_passed = False # Force fallback

    if l4q_enabled_passed and L4QQuantizedConv2d != nn.Conv2d:
        # print(f"DEBUG: Membuat L4QQuantizedConv2d ({in_channels}, {out_channels}, k={kernel_size})")
        if isinstance(kernel_size, int): # Pastikan kernel_size adalah tuple untuk L4QQuantizedConv2d
            kernel_size_tuple = (kernel_size, kernel_size)
        else:
            kernel_size_tuple = kernel_size
        return L4QQuantizedConv2d(
            in_channels, out_channels, kernel_size_tuple, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            lora_rank=lora_rank, n_bits=n_bits, alpha=alpha, 
            quant_group_size=quant_group_size
        )
    else:
        # print(f"DEBUG: Fallback ke nn.Conv{dims}d ({in_channels}, {out_channels}, k={kernel_size})")
        if dims == 1:
            return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        raise ValueError(f"Dimensi tidak didukung untuk conv standar: {dims}")

