import torch
import math # Diperlukan untuk math.log dan math.exp jika ada di l4q_init_scale, atau untuk sqrt

# File ini TIDAK BOLEH mengimpor dari l4q_linear_layer.py atau l4q_conv2d_layer.py
# atau l4q_utils.py (yang baru, yang akan menjadi factory)

def get_quantization_bounds(n_bits: int):
    """
    Calculates the lower (Q_N) and upper (Q_P) bounds for n-bit quantization.
    """
    q_n = -2**(n_bits - 1)
    q_p = 2**(n_bits - 1) - 1
    return q_n, q_p

def l4q_init_scale(weight_tensor: torch.Tensor, n_bits: int, group_size: int = -1):
    """
    Initializes the scale parameter (s) for quantization using the L4Q_init method.
    This method aims to minimize clipping error by using a conservative scale.
    Source: (referring to Equation 15 in the L4Q paper)

    Args:
        weight_tensor (torch.Tensor): The weight tensor to be quantized.
        n_bits (int): The number of bits for quantization.
        group_size (int): The group size for group-wise quantization. 
                          If -1, performs per-tensor quantization.

    Returns:
        torch.Tensor: The calculated scale parameter (s).
    """
    q_n, q_p = get_quantization_bounds(n_bits)

    if weight_tensor.numel() == 0: # Handle empty tensor
        return torch.tensor(1e-9, device=weight_tensor.device, dtype=weight_tensor.dtype)

    if group_size == -1 or weight_tensor.numel() <= group_size: # Per-tensor or if tensor is too small for grouping
        w_min = weight_tensor.min().item()
        w_max = weight_tensor.max().item()
        
        scale_n = abs(w_min / q_n) if q_n != 0 and w_min != 0 else float('inf')
        scale_p = abs(w_max / q_p) if q_p != 0 and w_max != 0 else float('inf')

        current_scale = 0.0
        if scale_n == float('inf') and scale_p == float('inf'): 
            current_scale = 1e-9 # Default for all-zero tensor
        elif scale_n == float('inf'):
            current_scale = scale_p
        elif scale_p == float('inf'):
            current_scale = scale_n
        else:
            current_scale = max(scale_n, scale_p)
        
        if current_scale == 0.0 or not math.isfinite(current_scale): # Prevent zero or non-finite scale
            current_scale = 1e-9
            
        return torch.tensor(current_scale, device=weight_tensor.device, dtype=weight_tensor.dtype)
    else: # Group-wise quantization
        if weight_tensor.numel() % group_size != 0:
            print(f"WARNING (l4q_init_scale): Weight tensor size ({weight_tensor.numel()}) "
                  f"is not divisible by group_size ({group_size}). "
                  f"Falling back to per-tensor quantization for this weight.")
            return l4q_init_scale(weight_tensor, n_bits, group_size=-1) # Fallback
        
        num_groups = weight_tensor.numel() // group_size
        # Ensure reshape is possible, should be guaranteed by the check above
        try:
            weight_groups = weight_tensor.reshape(num_groups, group_size)
        except RuntimeError as e:
            print(f"ERROR (l4q_init_scale): Could not reshape weight tensor for group-wise quantization. Tensor size: {weight_tensor.numel()}, groups: {num_groups}, group_size: {group_size}. Error: {e}")
            # Fallback to per-tensor if reshape fails unexpectedly
            return l4q_init_scale(weight_tensor, n_bits, group_size=-1)

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
            
            if current_scale_group == 0.0 or not math.isfinite(current_scale_group): # Prevent zero or non-finite scale
                current_scale_group = 1e-9
            scales[i] = current_scale_group
            
        return scales # Shape [num_groups]
