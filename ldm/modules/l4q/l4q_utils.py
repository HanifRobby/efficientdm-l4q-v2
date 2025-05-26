import torch.nn as nn # Diperlukan untuk fallback

# Impor layer L4Q dasar dari file masing-masing
try:
    from .l4q_linear_layer import L4QQuantizedLinear
    from .l4q_conv2d_layer import L4QQuantizedConv2d
    print("DEBUG l4q_utils.py (factory): Berhasil mengimpor L4QQuantizedLinear dan L4QQuantizedConv2d.")
except ImportError as e:
    print(f"PERINGATAN KRITIKAL di l4q_utils.py (factory): Gagal mengimpor L4QQuantizedLinear/Conv2d dari .l4q_..._layer.py. Error: {e}. Fungsi helper L4Q akan fallback ke nn.Linear/Conv2d.")
    L4QQuantizedLinear = nn.Linear 
    L4QQuantizedConv2d = nn.Conv2d # Ini akan menyebabkan masalah jika dims != 2 di fallback make_l4q_conv2d

# Fungsi inti (get_quantization_bounds, l4q_init_scale) TIDAK ADA LAGI DI SINI.
# Mereka ada di l4q_quant_core.py dan digunakan langsung oleh layer L4Q.

# --- Fungsi Helper L4Q (Factory) ---
def make_l4q_linear(in_features, out_features, bias=True, 
                    lora_rank=4, n_bits=4, alpha=1.0, group_size=-1, # group_size untuk L4QLinear
                    l4q_enabled_passed=True # Flag untuk kontrol eksternal
                    ):
    """Helper function to create L4QQuantizedLinear layer or fallback to nn.Linear."""
    if l4q_enabled_passed and L4QQuantizedLinear.__name__ != 'Linear': # Cek apakah fallback terjadi saat impor L4QQuantizedLinear
        # print(f"DEBUG l4q_utils (factory): Membuat L4QQuantizedLinear ({in_features}, {out_features})")
        return L4QQuantizedLinear(
            in_features=in_features, out_features=out_features, bias=bias,
            lora_rank=lora_rank, n_bits=n_bits, alpha=alpha, 
            group_size=group_size # Nama argumen di L4QQuantizedLinear adalah group_size
        )
    else:
        if l4q_enabled_passed and L4QQuantizedLinear.__name__ == 'Linear':
             print(f"INFO l4q_utils: L4QQuantizedLinear tidak terimpor dengan benar, fallback ke nn.Linear ({in_features}, {out_features}) meskipun l4q_enabled_passed=True.")
        # print(f"DEBUG l4q_utils (factory): Fallback ke nn.Linear ({in_features}, {out_features})")
        return nn.Linear(in_features, out_features, bias=bias)

def make_l4q_conv2d(dims, # 'dims' dari OpenAI UNet (1, 2, atau 3)
                    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                    lora_rank=4, n_bits=4, alpha=1.0, quant_group_size=-1, # quant_group_size untuk L4QConv2d
                    l4q_enabled_passed=True # Flag untuk kontrol eksternal
                    ):
    """Helper function to create L4QQuantizedConv2d layer or fallback to standard nn.ConvXd."""
    
    # L4QQuantizedConv2d kita saat ini hanya mendukung dims=2
    if dims != 2 and l4q_enabled_passed and L4QQuantizedConv2d.__name__ != 'Conv2d':
        print(f"PERINGATAN l4q_utils (factory): make_l4q_conv2d dipanggil dengan dims={dims} tapi L4QQuantizedConv2d hanya mendukung dims=2. Fallback ke nn.Conv{dims}d.")
        l4q_enabled_passed = False # Paksa fallback jika dims bukan 2 untuk L4Q

    if l4q_enabled_passed and L4QQuantizedConv2d.__name__ != 'Conv2d':
        # print(f"DEBUG l4q_utils (factory): Membuat L4QQuantizedConv2d ({in_channels}, {out_channels}, k={kernel_size})")
        if isinstance(kernel_size, int): 
            kernel_size_tuple = (kernel_size, kernel_size)
        else:
            kernel_size_tuple = kernel_size
        return L4QQuantizedConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_tuple, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            lora_rank=lora_rank, n_bits=n_bits, alpha=alpha, 
            quant_group_size=quant_group_size # Nama argumen di L4QQuantizedConv2d adalah quant_group_size
        )
    else:
        if l4q_enabled_passed and L4QQuantizedConv2d.__name__ == 'Conv2d':
            print(f"INFO l4q_utils: L4QQuantizedConv2d tidak terimpor dengan benar, fallback ke nn.Conv{dims}d ({in_channels}, {out_channels}) meskipun l4q_enabled_passed=True.")
        # print(f"DEBUG l4q_utils (factory): Fallback ke nn.Conv{dims}d ({in_channels}, {out_channels}, k={kernel_size})")
        if dims == 1:
            return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        raise ValueError(f"Dimensi tidak didukung untuk conv standar: {dims}")

