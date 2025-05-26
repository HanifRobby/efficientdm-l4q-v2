import torch

def get_quantization_bounds(n_bits: int):
    """
    Menghitung batas bawah (Q_N) dan batas atas (Q_P) untuk kuantisasi n-bit.
    Sumber: [cite: 47] (merujuk pada definisi Q_N dan Q_P di paper L4Q)
    """
    q_n = -2**(n_bits - 1)
    q_p = 2**(n_bits - 1) - 1
    return q_n, q_p

def l4q_init_scale(weight_tensor: torch.Tensor, n_bits: int, group_size: int = -1):
    """
    Menginisialisasi parameter skala (s) untuk kuantisasi menggunakan metode L4Q_init.
    Metode ini bertujuan meminimalkan clipping error dengan menggunakan skala konservatif.
    Sumber: [cite: 100] (merujuk pada Persamaan 15 di paper L4Q)

    Args:
        weight_tensor (torch.Tensor): Tensor bobot yang akan dikuantisasi.
        n_bits (int): Jumlah bit untuk kuantisasi.
        group_size (int): Ukuran grup untuk group-wise quantization. 
                          Jika -1, lakukan per-tensor quantization.
                          Paper L4Q menyebutkan group size[cite: 115].

    Returns:
        torch.Tensor: Parameter skala (s) yang dihitung.
    """
    q_n, q_p = get_quantization_bounds(n_bits)

    if group_size == -1: # Per-tensor quantization
        w_min = weight_tensor.min().item()
        w_max = weight_tensor.max().item()
        
        # Mencegah pembagian dengan nol jika q_n atau q_p adalah 0 (misalnya untuk n_bits=1 jika salah didefinisikan)
        # atau jika bobot semua nol.
        scale_n = abs(w_min / q_n) if q_n != 0 and w_min != 0 else float('inf')
        scale_p = abs(w_max / q_p) if q_p != 0 and w_max != 0 else float('inf')

        if scale_n == float('inf') and scale_p == float('inf'): # Jika semua bobot nol
            scale = torch.tensor(1e-9, device=weight_tensor.device, dtype=weight_tensor.dtype) # Skala kecil default
        elif scale_n == float('inf'):
            scale = scale_p
        elif scale_p == float('inf'):
            scale = scale_n
        else:
            scale = max(scale_n, scale_p)
        
        # Jika skala masih nol (misalnya w_min dan w_max sangat kecil), set ke nilai kecil
        if scale == 0:
            scale = 1e-9
            
        return torch.tensor(scale, device=weight_tensor.device, dtype=weight_tensor.dtype)
    else: # Group-wise quantization
        if weight_tensor.numel() % group_size != 0:
            raise ValueError("Ukuran tensor bobot harus dapat dibagi habis oleh group_size.")
        
        num_groups = weight_tensor.numel() // group_size
        weight_groups = weight_tensor.view(num_groups, group_size)
        
        scales = torch.zeros(num_groups, device=weight_tensor.device, dtype=weight_tensor.dtype)
        
        for i in range(num_groups):
            w_min_group = weight_groups[i].min().item()
            w_max_group = weight_groups[i].max().item()

            scale_n_group = abs(w_min_group / q_n) if q_n != 0 and w_min_group != 0 else float('inf')
            scale_p_group = abs(w_max_group / q_p) if q_p != 0 and w_max_group != 0 else float('inf')

            current_scale = 0
            if scale_n_group == float('inf') and scale_p_group == float('inf'):
                current_scale = 1e-9 
            elif scale_n_group == float('inf'):
                current_scale = scale_p_group
            elif scale_p_group == float('inf'):
                current_scale = scale_n_group
            else:
                current_scale = max(scale_n_group, scale_p_group)
            
            if current_scale == 0:
                current_scale = 1e-9
            scales[i] = current_scale
            
        return scales.view(-1, 1) # Sesuaikan shape jika diperlukan untuk broadcasting