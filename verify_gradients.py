import torch
from torch.autograd import gradcheck
import sys
import os

# Tambahkan path root proyek jika diperlukan agar impor modul L4Q berhasil
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ldm.modules.l4q.l4q_linear_layer import L4QQuantizedLinearFunction
    from ldm.modules.l4q.l4q_conv2d_layer import L4QQuantizedConv2dFunction
    print("Berhasil mengimpor L4QQuantizedLinearFunction dan L4QQuantizedConv2dFunction.")
except ImportError as e:
    print(f"Gagal mengimpor fungsi L4Q. Pastikan path sudah benar dan file ada. Error: {e}")
    print("Pastikan Anda menjalankan skrip ini dari direktori root proyek atau sys.path sudah diatur.")
    sys.exit(1)

dtype = torch.double
device = torch.device("cpu") 

def check_tensor_grad(grad_tensor, name="gradient"):
    """Helper untuk memeriksa validitas sebuah tensor gradien."""
    if grad_tensor is None:
        print(f"  {name}: None (OK jika input tidak memerlukan gradien atau bukan tensor)")
        return True
    if not isinstance(grad_tensor, torch.Tensor):
        print(f"  {name}: Bukan Tensor! Tipe: {type(grad_tensor)}")
        return False
    
    valid = True
    print(f"  {name}: shape={grad_tensor.shape}, dtype={grad_tensor.dtype}, device={grad_tensor.device}")
    if torch.isnan(grad_tensor).any():
        print(f"  WARNING: {name} mengandung NaN!")
        valid = False
    if torch.isinf(grad_tensor).any():
        print(f"  WARNING: {name} mengandung Inf!")
        valid = False
    if grad_tensor.dtype != dtype:
        print(f"  WARNING: {name} dtype ({grad_tensor.dtype}) tidak cocok dengan input dtype ({dtype})!")
        # valid = False # gradcheck bisa menangani ini, tapi baik untuk diketahui
    return valid


def run_gradcheck(func, inputs_tuple, function_name, eps=1e-6, atol=1e-5, rtol=1e-3):
    """Helper untuk menjalankan gradcheck dan mencetak hasilnya."""
    print(f"\n--- Menjalankan gradcheck untuk {function_name} ---")
    print("Status requires_grad untuk input Tensors:")
    for i, inp in enumerate(inputs_tuple):
        if isinstance(inp, torch.Tensor):
            print(f"  Input {i} (Tensor): requires_grad={inp.requires_grad}, dtype={inp.dtype}, shape={inp.shape}")
        else:
            print(f"  Input {i} (Non-Tensor): {type(inp)}, value={inp}")
    
    # Pastikan semua input tensor memiliki dtype yang benar
    processed_inputs = []
    for inp in inputs_tuple:
        if isinstance(inp, torch.Tensor):
            processed_inputs.append(inp.to(dtype=dtype, device=device))
        else:
            processed_inputs.append(inp)
    inputs_tuple_double = tuple(processed_inputs)

    try:
        # Jalankan dengan raise_exception=False untuk melihat perbedaan numerik
        test_passed = gradcheck(func, inputs_tuple_double, eps=eps, atol=atol, rtol=rtol, raise_exception=False, check_undefined_grad=True)
        if test_passed:
            print(f"GRADCHECK LULUS (secara numerik) untuk {function_name}")
        else:
            print(f"GRADCHECK GAGAL (secara numerik) untuk {function_name}. Periksa output di atas untuk perbedaan Jacobian.")
        return test_passed
    except RuntimeError as e: # Tangkap RuntimeError spesifik dari backward pass
        print(f"RuntimeError selama gradcheck untuk {function_name}: {e}")
        print("Ini biasanya menunjukkan masalah dalam fungsi backward() Anda, seperti return gradien yang salah.")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e: # Tangkap error lain
        print(f"Error umum selama gradcheck untuk {function_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_l4q_linear_grad():
    print("\n=== Pengujian L4QQuantizedLinearFunction ===")
    bs, in_feat, out_feat, r_val = 1, 3, 2, 2 
    n_b = 4 
    alpha_val = 1.0 
    
    x_lin = torch.randn(bs, in_feat, device=device).to(dtype).requires_grad_(True)
    w0_lin = torch.randn(out_feat, in_feat, device=device).to(dtype).requires_grad_(False)
    lora_a_lin = torch.randn(r_val, in_feat, device=device).to(dtype).requires_grad_(True)
    lora_b_lin = torch.randn(out_feat, r_val, device=device).to(dtype).requires_grad_(True)
    
    group_size_lin = -1 
    num_elements = out_feat * in_feat
    if group_size_lin != -1 and num_elements % group_size_lin == 0 and num_elements > 0:
        num_groups = num_elements // group_size_lin
        q_scale_lin_val = (torch.rand(num_groups, device=device) * 0.2 + 0.05).to(dtype).requires_grad_(True)
    else:
        if group_size_lin != -1: print(f"Linear: Fallback ke per-tensor q_scale untuk gradcheck (num_elements={num_elements}, group_size={group_size_lin})")
        q_scale_lin_val = torch.tensor([0.15], device=device).to(dtype).requires_grad_(True)
        group_size_lin = -1 # Pastikan konsisten

    inputs_lin = (x_lin, w0_lin, lora_a_lin, lora_b_lin, 
                  alpha_val, n_b, q_scale_lin_val, group_size_lin)
    
    run_gradcheck(L4QQuantizedLinearFunction.apply, inputs_lin, "L4QQuantizedLinearFunction", 
                  atol=1e-3, rtol=1e-2) # Toleransi dinaikkan

def test_l4q_conv2d_grad():
    print("\n=== Pengujian L4QQuantizedConv2dFunction ===")
    bs_conv, cin, cout, kh, kw, r_conv = 1, 2, 2, 2, 2, 1 # cout diperkecil untuk mengurangi elemen
    h_in, w_in = 3, 3 
    n_b_conv = 4
    alpha_conv = 1.0
    
    x_conv = torch.randn(bs_conv, cin, h_in, w_in, device=device).to(dtype).requires_grad_(True)
    w0_conv = torch.randn(cout, cin, kh, kw, device=device).to(dtype).requires_grad_(False)
    
    num_kernel_elements = cin * kh * kw
    lora_a_conv = torch.randn(r_conv, num_kernel_elements, device=device).to(dtype).requires_grad_(True)
    lora_b_conv = torch.randn(cout, r_conv, device=device).to(dtype).requires_grad_(True)
    bias_conv = torch.randn(cout, device=device).to(dtype).requires_grad_(True)
    # bias_conv = None # Jika tidak ada bias

    quant_group_size_conv = -1
    num_weight_elements = cout * cin * kh * kw
    if quant_group_size_conv != -1 and num_weight_elements % quant_group_size_conv == 0 and num_weight_elements > 0 :
        num_groups_conv = num_weight_elements // quant_group_size_conv
        q_scale_conv_val = (torch.rand(num_groups_conv, device=device) * 0.2 + 0.05).to(dtype).requires_grad_(True)
    else:
        if quant_group_size_conv != -1: print(f"Conv2d: Fallback ke per-tensor q_scale untuk gradcheck (num_elements={num_weight_elements}, group_size={quant_group_size_conv})")
        q_scale_conv_val = torch.tensor([0.15], device=device).to(dtype).requires_grad_(True)
        quant_group_size_conv = -1 # Pastikan konsisten

    stride_conv, padding_conv, dilation_conv, groups_conv = (1,1), (1,1), (1,1), 1 # Padding diubah agar output tidak terlalu kecil

    inputs_conv = (x_conv, w0_conv, lora_a_conv, lora_b_conv, bias_conv,
                   alpha_conv, n_b_conv, q_scale_conv_val, quant_group_size_conv,
                   stride_conv, padding_conv, dilation_conv, groups_conv)
    
    run_gradcheck(L4QQuantizedConv2dFunction.apply, inputs_conv, "L4QQuantizedConv2dFunction", 
                  atol=5e-3, rtol=5e-2) # Toleransi lebih tinggi untuk Conv2d

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA tersedia, tapi gradcheck akan dijalankan di CPU dengan double precision.")

    test_l4q_linear_grad()
    test_l4q_conv2d_grad()

    print("\n--- Semua pengujian gradcheck selesai ---")
