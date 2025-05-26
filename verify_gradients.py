import torch
from torch.autograd import gradcheck
import sys
import os

# Tambahkan path root proyek jika diperlukan
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ldm.modules.l4q.l4q_linear_layer import L4QQuantizedLinearFunction
    from ldm.modules.l4q.l4q_conv2d_layer import L4QQuantizedConv2dFunction
    print("Berhasil mengimpor L4QQuantizedLinearFunction dan L4QQuantizedConv2dFunction.")
except ImportError as e:
    print(f"Gagal mengimpor fungsi L4Q. Pastikan path sudah benar dan file ada. Error: {e}")
    sys.exit(1)

DTYPE = torch.double # Gunakan konstanta agar konsisten
DEVICE = torch.device("cpu") 

def run_gradcheck_wrapper(func, inputs_tuple, function_name, eps=1e-6, atol=1e-5, rtol=1e-3):
    """Wrapper untuk menjalankan gradcheck dan menangani output."""
    print(f"\n--- Menjalankan gradcheck untuk {function_name} ---")
    print("Status requires_grad untuk input Tensors:")
    processed_inputs = []
    for i, inp in enumerate(inputs_tuple):
        if isinstance(inp, torch.Tensor):
            # Pindahkan ke device dan dtype yang benar SEBELUM gradcheck
            inp_processed = inp.to(device=DEVICE, dtype=DTYPE)
            # Pastikan requires_grad diset dengan benar SETELAH konversi dtype jika diperlukan
            if inp.requires_grad: # Salin status requires_grad dari input asli
                inp_processed.requires_grad_()
            else:
                inp_processed.requires_grad_(False)
            processed_inputs.append(inp_processed)
            print(f"  Input {i} (Tensor): name={inp_name(inputs_tuple, i)}, requires_grad={inp_processed.requires_grad}, dtype={inp_processed.dtype}, shape={inp_processed.shape}")
        else:
            processed_inputs.append(inp)
            print(f"  Input {i} (Non-Tensor): name={inp_name(inputs_tuple, i)}, type={type(inp)}, value={inp}")
    
    inputs_tuple_double_device = tuple(processed_inputs)

    # Untuk melihat input yang sebenarnya masuk ke gradcheck
    # print("\nInputs yang diteruskan ke gradcheck:")
    # for i, p_inp in enumerate(inputs_tuple_double_device):
    #     if isinstance(p_inp, torch.Tensor):
    #         print(f"  Processed Input {i}: requires_grad={p_inp.requires_grad}, dtype={p_inp.dtype}, shape={p_inp.shape}, device={p_inp.device}")
    #     else:
    #         print(f"  Processed Input {i}: {type(p_inp)}")


    try:
        # check_undefined_grad=False bisa membantu jika ada gradien yang memang None dan gradcheck bingung
        # Namun, jika True, ia lebih ketat. Mari coba False dulu jika masih ada error []
        test_passed = gradcheck(func, inputs_tuple_double_device, eps=eps, atol=atol, rtol=rtol, 
                                raise_exception=False, check_analytical_jacobian_attributes=True,
                                check_undefined_grad=True) # Set True untuk lebih ketat
        if test_passed:
            print(f"GRADCHECK LULUS (secara numerik) untuk {function_name}")
        else:
            print(f"GRADCHECK GAGAL (secara numerik) untuk {function_name}. Periksa output di atas untuk perbedaan Jacobian atau pesan error.")
        return test_passed
    except RuntimeError as e:
        print(f"RuntimeError SELAMA gradcheck untuk {function_name}: {e}")
        print("Ini biasanya menunjukkan masalah dalam fungsi backward() Anda (misalnya, return gradien yang salah, operasi inplace).")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Error UMUM selama gradcheck untuk {function_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def inp_name(inputs_tuple, index):
    # Helper sederhana untuk memberi nama pada input berdasarkan urutan
    names_lin = ["x", "w0", "lora_a", "lora_b", "alpha", "n_bits", "q_scale", "group_size"]
    names_conv = ["x", "w0", "lora_a", "lora_b", "bias", "alpha", "n_bits", "q_scale", "quant_group_size", "stride", "padding", "dilation", "groups"]
    if len(inputs_tuple) == 8: # Linear
        return names_lin[index] if index < len(names_lin) else f"input_{index}"
    elif len(inputs_tuple) == 13: # Conv2d
        return names_conv[index] if index < len(names_conv) else f"input_{index}"
    return f"input_{index}"


def test_l4q_linear_grad():
    print("\n=== Pengujian L4QQuantizedLinearFunction ===")
    bs, in_feat, out_feat, r_val = 1, 3, 2, 2 
    n_b = 4 
    alpha_val = 1.0 
    
    x_lin = torch.randn(bs, in_feat, device=DEVICE).requires_grad_(True) # dtype akan diatur di run_gradcheck_wrapper
    w0_lin = torch.randn(out_feat, in_feat, device=DEVICE).requires_grad_(False)
    lora_a_lin = torch.randn(r_val, in_feat, device=DEVICE).requires_grad_(True)
    lora_b_lin = torch.randn(out_feat, r_val, device=DEVICE).requires_grad_(True)
    
    group_size_lin = -1 
    num_elements = out_feat * in_feat
    q_scale_lin_val_init = None
    if group_size_lin != -1 and num_elements % group_size_lin == 0 and num_elements > 0:
        num_groups = num_elements // group_size_lin
        q_scale_lin_val_init = (torch.rand(num_groups, device=DEVICE) * 0.2 + 0.05) # Inisialisasi dengan nilai positif
    else:
        if group_size_lin != -1: print(f"Linear: Fallback ke per-tensor q_scale untuk gradcheck (num_elements={num_elements}, group_size={group_size_lin})")
        q_scale_lin_val_init = torch.tensor([0.15], device=DEVICE)
        group_size_lin = -1 
    q_scale_lin_val = q_scale_lin_val_init.clone().detach().requires_grad_(True)


    inputs_lin = (x_lin, w0_lin, lora_a_lin, lora_b_lin, 
                  alpha_val, n_b, q_scale_lin_val, group_size_lin)
    
    return run_gradcheck_wrapper(L4QQuantizedLinearFunction.apply, inputs_lin, "L4QQuantizedLinearFunction", 
                                 atol=1e-3, rtol=1e-2) # Toleransi dinaikkan

def test_l4q_conv2d_grad():
    print("\n=== Pengujian L4QQuantizedConv2dFunction ===")
    bs_conv, cin, cout, kh, kw, r_conv = 1, 2, 2, 2, 2, 1 
    h_in, w_in = 3, 3 
    n_b_conv = 4
    alpha_conv = 1.0
    
    x_conv = torch.randn(bs_conv, cin, h_in, w_in, device=DEVICE).requires_grad_(True)
    w0_conv = torch.randn(cout, cin, kh, kw, device=DEVICE).requires_grad_(False)
    
    num_kernel_elements = cin * kh * kw
    lora_a_conv = torch.randn(r_conv, num_kernel_elements, device=DEVICE).requires_grad_(True)
    lora_b_conv = torch.randn(cout, r_conv, device=DEVICE).requires_grad_(True)
    bias_conv = torch.randn(cout, device=DEVICE).requires_grad_(True)
    # bias_conv = None

    quant_group_size_conv = -1
    num_weight_elements = cout * cin * kh * kw
    q_scale_conv_val_init = None
    if quant_group_size_conv != -1 and num_weight_elements % quant_group_size_conv == 0 and num_weight_elements > 0 :
        num_groups_conv = num_weight_elements // quant_group_size_conv
        q_scale_conv_val_init = (torch.rand(num_groups_conv, device=DEVICE) * 0.2 + 0.05)
    else:
        if quant_group_size_conv != -1: print(f"Conv2d: Fallback ke per-tensor q_scale untuk gradcheck (num_elements={num_weight_elements}, group_size={quant_group_size_conv})")
        q_scale_conv_val_init = torch.tensor([0.15], device=DEVICE)
        quant_group_size_conv = -1 
    q_scale_conv_val = q_scale_conv_val_init.clone().detach().requires_grad_(True)


    stride_conv, padding_conv, dilation_conv, groups_conv = (1,1), (1,1), (1,1), 1

    inputs_conv = (x_conv, w0_conv, lora_a_conv, lora_b_conv, bias_conv,
                   alpha_conv, n_b_conv, q_scale_conv_val, quant_group_size_conv,
                   stride_conv, padding_conv, dilation_conv, groups_conv)
    
    return run_gradcheck_wrapper(L4QQuantizedConv2dFunction.apply, inputs_conv, "L4QQuantizedConv2dFunction", 
                                 atol=5e-3, rtol=5e-2)

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA tersedia, tapi gradcheck akan dijalankan di CPU dengan double precision.")

    passed_linear = test_l4q_linear_grad()
    passed_conv = test_l4q_conv2d_grad()

    print("\n--- Ringkasan Gradcheck ---")
    print(f"L4QQuantizedLinearFunction: {'LULUS' if passed_linear else 'GAGAL'}")
    print(f"L4QQuantizedConv2dFunction: {'LULUS' if passed_conv else 'GAGAL'}")
    print("--- Semua pengujian gradcheck selesai ---")
