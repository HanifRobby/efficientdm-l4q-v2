import sys
import os
import torch
from omegaconf import OmegaConf

# Tambahkan path root proyek ke sys.path jika skrip ini dijalankan dari direktori lain
# atau jika Anda menjalankan dari root, ini mungkin tidak selalu diperlukan
# tapi bisa membantu memastikan modul ditemukan.
# Sesuaikan jika struktur Anda berbeda atau Anda menggunakan PYTHONPATH.
# sys.path.append(os.getcwd()) # Menambahkan direktori kerja saat ini

# Impor kelas model utama dari LDM.
# Pastikan path impor ini sesuai dengan struktur proyek Anda.
try:
    from ldm.models.diffusion.ddpm import LatentDiffusion
    from ldm.util import instantiate_from_config # Diperlukan jika target di YAML adalah string
except ImportError as e:
    print(f"Gagal mengimpor modul LDM. Pastikan Anda menjalankan skrip dari direktori root proyek")
    print(f"dan bahwa PYTHONPATH Anda sudah benar jika diperlukan.")
    print(f"Error: {e}")
    sys.exit(1)

def main():
    # Ganti dengan path yang benar ke file konfigurasi L4Q Anda
    config_path = "configs/l4q_configs/l4q_cin256_v2.yaml" 

    print(f"Memuat konfigurasi dari: {config_path}")
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Gagal memuat file konfigurasi: {e}")
        return

    print("Konfigurasi berhasil dimuat.")
    print("Mencoba menginisialisasi model LatentDiffusion...")

    try:
        # Cara 1: Jika target di YAML adalah string path ke kelas
        # model = instantiate_from_config(config.model)

        # Cara 2: Jika Anda tahu kelasnya secara langsung dan YAML hanya berisi params
        # Ini lebih umum jika 'target' di YAML menunjuk ke kelas LatentDiffusion
        if config.model.target == "ldm.models.diffusion.ddpm.LatentDiffusion":
             # Pastikan semua parameter yang dibutuhkan oleh LatentDiffusion.__init__ ada di config.model.params
            model = LatentDiffusion(**config.model.params)
        else:
            # Jika target adalah path string, gunakan instantiate_from_config
            print(f"Menggunakan instantiate_from_config untuk target: {config.model.target}")
            model = instantiate_from_config(config.model)

        print("-" * 50)
        print(">>> Model BERHASIL diinisialisasi! <<<")
        print("-" * 50)
        
        # Mencetak keseluruhan struktur model.
        # Jika layer L4Q Anda memiliki implementasi extra_repr yang baik,
        # Anda akan melihat detail parameter L4Q di sini.
        print("\nStruktur Model:")
        print(model)
        
        # Anda juga bisa mencoba mengakses Unet secara spesifik jika path-nya diketahui
        # (ini bergantung pada bagaimana LatentDiffusion menyimpan Unet-nya)
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            unet_model = model.model.diffusion_model
            print("\nStruktur UNet (model.model.diffusion_model):")
            print(unet_model)
            print("\nParameter L4Q yang dikonfigurasi untuk Unet (jika ada):")
            if hasattr(unet_model, 'l4q_params'):
                print(unet_model.l4q_params)
            else:
                print("Atribut 'l4q_params' tidak ditemukan di Unet.")

            # Periksa apakah layer pertama di Unet (conv_in) adalah L4Q atau standar
            if hasattr(unet_model, 'conv_in'):
                print(f"\nLayer conv_in Unet: {type(unet_model.conv_in)}")
                if hasattr(unet_model.conv_in, 'extra_repr'):
                    print(f"Detail conv_in: {unet_model.conv_in.extra_repr()}")

        elif hasattr(model, 'diffusion_model'): # Struktur lain yang mungkin
            unet_model = model.diffusion_model
            print("\nStruktur UNet (model.diffusion_model):")
            print(unet_model)
            if hasattr(unet_model, 'l4q_params'):
                print(f"Parameter L4Q Unet: {unet_model.l4q_params}")


    except Exception as e:
        print("-" * 50)
        print(">>> GAGAL menginisialisasi model! <<<")
        print("-" * 50)
        print(f"Error: {e}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Pindah ke direktori root proyek jika perlu, agar path relatif bekerja
    # Ini mungkin tidak diperlukan jika Anda menjalankan dari root
    # current_script_path = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_script_path, "..")) # Asumsi skrip ada di subdir
    # os.chdir(project_root)
    # print(f"Direktori kerja saat ini: {os.getcwd()}")

    main()