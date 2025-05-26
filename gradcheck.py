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