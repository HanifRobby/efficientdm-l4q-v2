import torch
from torch.autograd import gradcheck
from ldm.modules.l4q.l4q_linear_layer import L4QQuantizedLinearFunction # Sesuaikan path
from ldm.modules.l4q.l4q_conv2d_layer import L4QQuantizedConv2dFunction # Sesuaikan path

# Pengujian untuk L4QQuantizedLinearFunction
print("Menguji L4QQuantizedLinearFunction...")
# Parameter dummy (buat sekecil mungkin untuk gradcheck)
in_feat, out_feat, r_val, n_b, bs = 2, 3, 1, 4, 1
x_lin = torch.randn(bs, in_feat, dtype=torch.double, requires_grad=True)
w0_lin = torch.randn(out_feat, in_feat, dtype=torch.double, requires_grad=False) # W0 tidak dilatih
la_lin = torch.randn(r_val, in_feat, dtype=torch.double, requires_grad=True)
lb_lin = torch.randn(out_feat, r_val, dtype=torch.double, requires_grad=True)
alpha_lin = 1.0
# q_scale perlu requires_grad=True jika ingin dilatih dan dicek gradiennya
# Untuk gradcheck, kita perlu memastikan q_scale tidak terlalu kecil atau menyebabkan instabilitas
# Mari kita inisialisasi secara manual untuk pengujian
q_scale_lin_val = torch.tensor([0.1], dtype=torch.double, requires_grad=True) 
quant_group_size_lin = -1 # Per-tensor
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

# Gunakan torch.double untuk presisi yang lebih tinggi saat gradcheck
dtype = torch.double
device = torch.device("cpu") # Gradcheck biasanya lebih mudah di-debug di CPU

def run_gradcheck(func, inputs_tuple, function_name, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False):
    """Helper untuk menjalankan gradcheck dan mencetak hasilnya."""
    print(f"\n--- Menjalankan gradcheck untuk {function_name} ---")
    # Periksa apakah semua input yang memerlukan gradien diset dengan benar
    print("Input requires_grad status:")
    for i, inp in enumerate(inputs_tuple):
        if isinstance(inp, torch.Tensor):
            print(f"  Input {i}: {inp.requires_grad}")
        else:
            print(f"  Input {i}: (Bukan Tensor)")

    try:
        test_passed = gradcheck(func, inputs_tuple, eps=eps, atol=atol, rtol=rtol, raise_exception=raise_exception)
        if test_passed:
            print(f"GRADCHECK LULUS untuk {function_name}")
        else:
            print(f"GRADCHECK GAGAL untuk {function_name}. Periksa output di atas untuk detail.")
        return test_passed
    except Exception as e:
        print(f"Error selama gradcheck untuk {function_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_l4q_linear_grad():
    print("\n=== Pengujian L4QQuantizedLinearFunction ===")
    # Parameter dummy (buat sekecil mungkin untuk gradcheck agar cepat)
    # Batch size, input features, output features, lora rank
    bs, in_feat, out_feat, r_val = 1, 3, 2, 2 
    n_b = 4 # bits
    alpha_val = 1.0
    
    # Input x
    x_lin = torch.randn(bs, in_feat, dtype=dtype, device=device, requires_grad=True)
    
    # Bobot asli W0 (tidak dilatih, tidak memerlukan gradien)
    w0_lin = torch.randn(out_feat, in_feat, dtype=dtype, device=device, requires_grad=False)
    
    # Parameter LoRA A dan B (dilatih)
    lora_a_lin = torch.randn(r_val, in_feat, dtype=dtype, device=device, requires_grad=True)
    lora_b_lin = torch.randn(out_feat, r_val, dtype=dtype, device=device, requires_grad=True)
    
    # Parameter Kuantisasi (skala 's') (dilatih)
    # Inisialisasi q_scale agar tidak terlalu dekat dengan nol untuk stabilitas numerik gradcheck
    q_scale_lin_val = torch.tensor([0.1] + [0.1 + 0.01*i for i in range( (in_feat*out_feat)//64 -1 if (in_feat*out_feat)//64 > 0 and (in_feat*out_feat)%64==0 else 0)], dtype=dtype, device=device, requires_grad=True) # Contoh untuk group_size=64
    # Jika group_size = -1 (per-tensor)
    # q_scale_lin_val = torch.tensor([0.1], dtype=dtype, device=device, requires_grad=True)


    # group_size: -1 untuk per-tensor, atau nilai positif untuk group-wise
    group_size_lin = -1 
    # Jika group_size_lin diubah menjadi > 0, pastikan q_scale_lin_val memiliki shape yang sesuai (num_groups)
    # Misalnya, jika group_size_lin = 3 (in_feat*out_feat / 3), maka q_scale_lin_val harus punya (in_feat*out_feat / 3) elemen.
    # Untuk contoh ini, in_feat*out_feat = 3*2 = 6. Jika group_size_lin = 3, num_groups = 2.
    # q_scale_lin_val = torch.tensor([0.1, 0.12], dtype=dtype, device=device, requires_grad=True)

    if group_size_lin != -1:
        num_elements = out_feat * in_feat
        if num_elements % group_size_lin == 0:
            num_groups = num_elements // group_size_lin
            q_scale_lin_val = torch.rand(num_groups, dtype=dtype, device=device) * 0.1 + 0.05 # Nilai acak kecil positif
            q_scale_lin_val.requires_grad_(True)
        else:
            print(f"Linear: num_elements {num_elements} tidak habis dibagi group_size {group_size_lin}, menggunakan per-tensor q_scale untuk gradcheck.")
            q_scale_lin_val = torch.tensor([0.1], dtype=dtype, device=device, requires_grad=True)
            group_size_lin = -1 # Fallback ke per-tensor jika tidak bisa dibagi
    else: # Per-tensor
        q_scale_lin_val = torch.tensor([0.1], dtype=dtype, device=device, requires_grad=True)


    inputs_lin = (x_lin, w0_lin, lora_a_lin, lora_b_lin, 
                  alpha_val, n_b, q_scale_lin_val, group_size_lin)
    
    # Cek gradien untuk semua input yang requires_grad=True
    # `nondet_tol` mungkin diperlukan jika ada operasi non-deterministik (jarang untuk ini)
    # `check_undefined_grad=True` (default) akan error jika ada gradien undefined yang tidak seharusnya.
    # `fast_mode=True` bisa mempercepat tapi kurang akurat.
    run_gradcheck(L4QQuantizedLinearFunction.apply, inputs_lin, "L4QQuantizedLinearFunction", 
                  atol=1e-4, rtol=1e-2, raise_exception=True) # Naikkan toleransi jika gagal karena presisi

def test_l4q_conv2d_grad():
    print("\n=== Pengujian L4QQuantizedConv2dFunction ===")
    # Parameter dummy
    bs_conv, cin, cout, kh, kw, r_conv = 1, 2, 3, 2, 2, 1 
    h_in, w_in = 4, 4 # Ukuran input spasial
    n_b_conv = 4
    alpha_conv = 1.0
    
    x_conv = torch.randn(bs_conv, cin, h_in, w_in, dtype=dtype, device=device, requires_grad=True)
    w0_conv = torch.randn(cout, cin, kh, kw, dtype=dtype, device=device, requires_grad=False)
    
    num_kernel_elements = cin * kh * kw
    lora_a_conv = torch.randn(r_conv, num_kernel_elements, dtype=dtype, device=device, requires_grad=True)
    lora_b_conv = torch.randn(cout, r_conv, dtype=dtype, device=device, requires_grad=True)
    
    bias_conv = torch.randn(cout, dtype=dtype, device=device, requires_grad=True)
    # bias_conv = None # Jika tidak ada bias

    quant_group_size_conv = -1 
    # Jika quant_group_size_conv > 0, sesuaikan q_scale_conv_val
    # num_weight_elements = cout * cin * kh * kw
    # if quant_group_size_conv != -1 and num_weight_elements % quant_group_size_conv == 0:
    #     num_groups_conv = num_weight_elements // quant_group_size_conv
    #     q_scale_conv_val = torch.rand(num_groups_conv, dtype=dtype, device=device) * 0.1 + 0.05
    #     q_scale_conv_val.requires_grad_(True)
    # else:
    #     q_scale_conv_val = torch.tensor([0.1], dtype=dtype, device=device, requires_grad=True)
    #     if quant_group_size_conv != -1: quant_group_size_conv = -1 # Fallback

    num_weight_elements = cout * cin * kh * kw
    if quant_group_size_conv != -1:
        if num_weight_elements % quant_group_size_conv == 0 and num_weight_elements > 0:
            num_groups_conv = num_weight_elements // quant_group_size_conv
            q_scale_conv_val = torch.rand(num_groups_conv, dtype=dtype, device=device) * 0.1 + 0.05 
            q_scale_conv_val.requires_grad_(True)
        else:
            print(f"Conv2d: num_elements {num_weight_elements} tidak habis dibagi group_size {quant_group_size_conv}, menggunakan per-tensor q_scale untuk gradcheck.")
            q_scale_conv_val = torch.tensor([0.1], dtype=dtype, device=device, requires_grad=True)
            quant_group_size_conv = -1 # Fallback
    else: # Per-tensor
        q_scale_conv_val = torch.tensor([0.1], dtype=dtype, device=device, requires_grad=True)

    stride_conv, padding_conv, dilation_conv, groups_conv = (1,1), (0,0), (1,1), 1

    inputs_conv = (x_conv, w0_conv, lora_a_conv, lora_b_conv, bias_conv,
                   alpha_conv, n_b_conv, q_scale_conv_val, quant_group_size_conv,
                   stride_conv, padding_conv, dilation_conv, groups_conv)
    
    # Gradcheck untuk Conv2d bisa lebih sensitif dan lambat
    run_gradcheck(L4QQuantizedConv2dFunction.apply, inputs_conv, "L4QQuantizedConv2dFunction", 
                  atol=1e-3, rtol=1e-2, raise_exception=True) # Toleransi mungkin perlu lebih tinggi

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    # Pastikan CUDA tidak dipakai jika tidak sengaja
    # karena gradcheck lebih stabil di CPU dan double precision
    if torch.cuda.is_available():
        print(f"CUDA tersedia, tapi gradcheck akan dijalankan di CPU dengan double precision.")

    test_l4q_linear_grad()
    test_l4q_conv2d_grad()

    print("\n--- Semua pengujian gradcheck selesai ---")

inputs_lin = (x_lin, w0_lin, la_lin, lb_lin, alpha_lin, n_b, q_scale_lin_val, quant_group_size_lin)
test_lin = gradcheck(L4QQuantizedLinearFunction.apply, inputs_lin, eps=1e-6, atol=1e-2, rtol=1e-2)
print("Tes L4QQuantizedLinearFunction lulus:", test_lin)


# Pengujian untuk L4QQuantizedConv2dFunction
print("\nMenguji L4QQuantizedConv2dFunction...")
# Parameter dummy
cin, cout, kh, kw, r_conv, n_b_conv, bs_conv = 2, 3, 2, 2, 1, 4, 1
h_in, w_in = 4, 4
x_conv = torch.randn(bs_conv, cin, h_in, w_in, dtype=torch.double, requires_grad=True)
w0_conv = torch.randn(cout, cin, kh, kw, dtype=torch.double, requires_grad=False)
# lora_a_conv: [r, cin * kh * kw], lora_b_conv: [cout, r]
la_conv = torch.randn(r_conv, cin * kh * kw, dtype=torch.double, requires_grad=True)
lb_conv = torch.randn(cout, r_conv, dtype=torch.double, requires_grad=True)
bias_conv = torch.randn(cout, dtype=torch.double, requires_grad=True) # Bias bisa dicek juga
alpha_conv = 1.0
q_scale_conv_val = torch.tensor([0.1], dtype=torch.double, requires_grad=True)
quant_group_size_conv = -1
stride_conv, padding_conv, dilation_conv, groups_conv = (1,1), (0,0), (1,1), 1

inputs_conv = (x_conv, w0_conv, la_conv, lb_conv, bias_conv, alpha_conv, n_b_conv,
               q_scale_conv_val, quant_group_size_conv,
               stride_conv, padding_conv, dilation_conv, groups_conv)
# Note: gradcheck mungkin lambat untuk conv, dan mungkin perlu atol/rtol yang lebih longgar
test_conv = gradcheck(L4QQuantizedConv2dFunction.apply, inputs_conv, eps=1e-6, atol=5e-2, rtol=5e-2)
print("Tes L4QQuantizedConv2dFunction lulus:", test_conv)