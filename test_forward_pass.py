import torch
from omegaconf import OmegaConf
import sys
import os

# Tambahkan path root proyek jika diperlukan
# sys.path.append(os.getcwd())

try:
    from ldm.models.diffusion.ddpm import LatentDiffusion
    from ldm.util import instantiate_from_config # Jika target di YAML adalah string
except ImportError as e:
    print(f"Gagal mengimpor modul LDM. Pastikan Anda menjalankan skrip dari direktori root proyek.")
    print(f"Error: {e}")
    sys.exit(1)

def main():
    config_path = "configs/l4q_configs/l4q_cin256_v2.yaml" # Sesuaikan path jika perlu

    print(f"Memuat konfigurasi dari: {config_path}")
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Gagal memuat file konfigurasi: {e}")
        return

    print("Konfigurasi berhasil dimuat.")
    
    # Tentukan device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    model = None # Inisialisasi model
    try:
        print("Mencoba menginisialisasi model LatentDiffusion...")
        if config.model.target == "ldm.models.diffusion.ddpm.LatentDiffusion":
            model = LatentDiffusion(**config.model.params)
        else:
            model = instantiate_from_config(config.model)
        
        model.to(device)
        model.eval() # Set model ke mode evaluasi
        print("Model berhasil diinisialisasi dan dipindahkan ke device.")

    except Exception as e:
        print(f"Gagal menginisialisasi model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Melakukan Uji Forward Pass Sederhana ---")
    
    # Dapatkan parameter dari konfigurasi untuk membuat input dummy
    # Ini adalah parameter untuk UNet, bukan untuk VAE atau gambar asli
    # Kita perlu tahu dimensi input latent yang diharapkan oleh UNet
    # Biasanya z_channels dari VAE dan image_size Unet (setelah VAE)
    
    # Asumsi dari config l4q_cin256_v2.yaml:
    # VAE (first_stage_config) ddconfig.z_channels = 3
    # Unet (unet_config) params.in_channels = 3
    # Unet (unet_config) params.image_size = 64 (ini adalah ukuran latent, bukan gambar asli 256)
    # Conditioning (cond_stage_config) embed_dim = 512 (ini akan menjadi context_dim Unet)
    
    batch_size = 2 # Coba dengan batch size > 1
    unet_in_channels = config.model.params.unet_config.params.get("in_channels", 3)
    # image_size di unet_config adalah ukuran latent, bukan gambar asli
    latent_size = config.model.params.unet_config.params.get("image_size", 64) 
    
    # Jika model menggunakan context (cross-attention)
    context_dim = config.model.params.unet_config.params.get("context_dim", None)
    # Jika model class-conditional
    num_classes = config.model.params.cond_stage_config.params.get("n_classes", None) if hasattr(config.model.params.cond_stage_config, "params") else None


    print(f"Membuat data dummy: batch_size={batch_size}, unet_in_channels={unet_in_channels}, latent_size={latent_size}x{latent_size}")
    if context_dim:
        print(f"Context dim: {context_dim}")
    if num_classes:
        print(f"Num classes: {num_classes}")

    try:
        # Input latent dummy untuk Unet
        dummy_x_latent = torch.randn(batch_size, unet_in_channels, latent_size, latent_size, device=device)
        
        # Timestep dummy
        dummy_timesteps = torch.randint(0, config.model.params.get("timesteps", 1000), (batch_size,), device=device).long()
        
        # Context dummy (jika diperlukan oleh model)
        dummy_context = None
        if context_dim:
            # Ukuran sequence untuk context biasanya 77 untuk CLIP
            # Jika model Anda menggunakan context_dim yang berbeda atau sumber context lain, sesuaikan ini.
            # Untuk class conditioning, context mungkin tidak dipakai jika y dipakai.
            # Jika conditioning_key: crossattn dan targetnya adalah CLIPTextEmbedder atau serupa
            dummy_context_sequence_length = 77 
            dummy_context = torch.randn(batch_size, dummy_context_sequence_length, context_dim, device=device)
            print(f"Menggunakan dummy context shape: {dummy_context.shape}")

        # Label kelas dummy (jika class-conditional)
        dummy_y = None
        if num_classes:
            dummy_y = torch.randint(0, num_classes, (batch_size,), device=device).long()
            print(f"Menggunakan dummy class labels (y) shape: {dummy_y.shape}")

        # Lakukan forward pass
        # Panggil metode 'apply_model' jika itu yang dipanggil selama training loop DDPM,
        # atau panggil 'model.forward' atau 'model.model.diffusion_model.forward'
        # tergantung pada bagaimana model Anda dirancang untuk dipanggil.
        # Untuk LatentDiffusion, biasanya kita ingin output dari Unet (noise prediction).
        # Metode `model.apply_model(x_noisy, t, cond, ...)` adalah yang umum.
        
        print("Mencoba memanggil model.apply_model(...) atau model.model.diffusion_model(...)")
        
        # Siapkan conditioning dictionary seperti yang diharapkan LDM
        conditioning_kwargs = {}
        if dummy_context is not None and model.cond_stage_key == "crossattn": # atau sesuai cond_stage_key Anda
            # LDM biasanya mengharapkan context dalam list di dalam dict
            # c_crossattn adalah key yang umum
            conditioning_kwargs['cond'] = {"c_crossattn": [dummy_context]} 
        elif dummy_y is not None and model.cond_stage_key == "class_label":
            conditioning_kwargs['cond'] = {"c_concat": [model.get_learned_conditioning(dummy_y.unsqueeze(1))]} # Contoh jika class label perlu di-embed dulu
            # Atau jika model.apply_model langsung menerima y:
            # conditioning_kwargs['y'] = dummy_y # Ini kurang umum untuk apply_model LDM

        # Cek apakah model.apply_model ada
        if hasattr(model, 'apply_model') and callable(model.apply_model):
            # Argumen untuk apply_model biasanya x_noisy, t, cond
            # `cond` bisa berupa dict yang berisi context atau class embeddings
            # Untuk class conditioning, LDM biasanya menggunakan `get_learned_conditioning`
            # dan meneruskannya sebagai bagian dari `cond` atau argumen `y`.
            
            # Buat conditioning dictionary `c`
            c = {}
            if dummy_context is not None and "crossattn" in model.conditioning_key:
                 # LDM seringkali mengharapkan context dalam list
                c[model.cond_stage_key] = [dummy_context] 
            
            if dummy_y is not None and "class_label" in model.cond_stage_key:
                # Jika model Anda langsung menggunakan y di forward Unet (melalui label_emb)
                # maka y perlu diproses oleh cond_stage_model terlebih dahulu.
                # Atau, jika Unet langsung menerima y untuk nn.Embedding.
                # Untuk LDM, biasanya `get_learned_conditioning` dipanggil.
                # Mari kita asumsikan `y` bisa langsung dipakai atau perlu diproses.
                # Jika cond_stage_config adalah ClassEmbedder, ia akan dipanggil oleh LatentDiffusion
                # untuk menghasilkan embedding yang kemudian dimasukkan ke `c`.
                # Untuk pengujian sederhana, kita bisa mencoba meneruskan y jika ada.
                # Namun, LatentDiffusion.apply_model biasanya menerima `cond` bukan `y`.
                # Mari kita coba buat 'cond' yang sesuai.
                if model.cond_stage_model is not None:
                    # Ini akan menghasilkan embedding kelas
                    class_cond = model.get_learned_conditioning(dummy_y) 
                    if isinstance(class_cond, dict): # Jika get_learned_conditioning mengembalikan dict
                        c.update(class_cond)
                    else: # Asumsikan itu adalah tensor embedding
                        c[model.cond_stage_key] = class_cond # atau c_concat, c_vector
                else: # Jika tidak ada cond_stage_model eksplisit, mungkin Unet menangani y
                    pass # Biarkan y ditangani oleh **conditioning_kwargs jika ada

            print(f"Conditioning dict 'c' untuk apply_model: { {k: v.shape if isinstance(v, torch.Tensor) else (type(v[0]) if isinstance(v,list) else type(v)) for k,v in c.items()} }")

            # Panggil apply_model jika ada
            with torch.no_grad(): # Tidak perlu gradien untuk forward pass test
                output_noise_pred = model.apply_model(dummy_x_latent, dummy_timesteps, c)
            print(f"Forward pass (model.apply_model) berhasil! Output shape: {output_noise_pred.shape}")
            assert output_noise_pred.shape == dummy_x_latent.shape, "Output shape tidak cocok dengan input shape!"

        elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'): # UNet
            unet = model.model.diffusion_model
            print("Mencoba memanggil Unet secara langsung (model.model.diffusion_model.forward)...")
            with torch.no_grad():
                # Unet OpenAI biasanya menerima x, timesteps, context (opsional), y (opsional)
                # Unet LDM (Model di model.py) menerima x, t, context (untuk SpatialTransformer)
                # Pastikan argumen sesuai dengan definisi forward Unet Anda.
                if hasattr(unet, 'l4q_params'): # Jika UNet kita adalah yang dari openaimodel.py
                    # UNetModel OpenAI menerima (x, timesteps, context=None, y=None)
                    # 'context' di sini untuk SpatialTransformer, 'y' untuk class conditioning
                    output_noise_pred = unet(dummy_x_latent, dummy_timesteps, context=dummy_context, y=dummy_y)
                else: # Jika UNet kita adalah kelas Model dari model.py
                    output_noise_pred = unet(dummy_x_latent, dummy_timesteps, context=dummy_context) # Context di sini untuk SpatialTransformer
            print(f"Forward pass (Unet langsung) berhasil! Output shape: {output_noise_pred.shape}")
            assert output_noise_pred.shape == dummy_x_latent.shape, "Output shape tidak cocok dengan input shape!"
        else:
            print("Tidak dapat menemukan metode forward yang sesuai (apply_model atau diffusion_model).")


    except Exception as e:
        print(f"Gagal melakukan forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
