"""
Convention

ours/naive-fft-(fft_h-fft_w)-learn-(learn_h-learn_w)-meas-(meas_h-meas-w)-kwargs

* Phlatcam: 1518 x 2012 (post demosiacking)
"""
from pathlib import Path
import torch
from types import SimpleNamespace

# Define FFT arguments once at the module level

height = 270
width = 480
fft_args_dict = {
    "psf_mat": Path("data/diffusercam/psf.tiff"),
    "psf_height": height,
    "psf_width": width,
    "psf_centre_x": height // 2,
    "psf_centre_y": width // 2,
    "psf_crop_size_x": height,
    "psf_crop_size_y": width,
    "meas_height": height,
    "meas_width": width,
    "meas_centre_x": height // 2,
    "meas_centre_y": width // 2,
    "meas_crop_size_x": height,
    "meas_crop_size_y": width,
    "pad_meas_mode": "replicate",
    "image_height": 270,
    "image_width": 480,
    "fft_gamma": 100,  # Gamma for Weiner init
    "fft_requires_grad": False,
    "fft_epochs": 0,
}

def base_config():
    exp_name = "fft-diffusercam"
    is_naive = "naive" in exp_name
    multi = 1
    use_spatial_weight = False
    weight_update = True
    dataset = "diffusercam"
    # Use FFT arguments from the global definition
    locals().update(fft_args_dict)
    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data/diffusercam")
    output_dir = Path("output/diffusercam") / exp_name
    ckpt_dir = Path("ckpts/diffusercam") / exp_name
    run_dir = Path("runs/diffusercam") / exp_name  # Tensorboard

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #


    shuffle = True
    train_gaussian_noise = 5e-3

    
    model = "UNet270480"
    batch_size = 18
    num_threads = batch_size >> 1  # parallel workers

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    num_epochs = 100
    fft_epochs = num_epochs if is_naive else 0

    learning_rate = 1e-4
    fft_learning_rate = 3e-5 

    # Betas for AdamW. We follow https://arxiv.org/pdf/1704.00028
    beta_1 = 0.9  # momentum
    beta_2 = 0.999

    lr_scheduler = "cosine"  # or step

    # Cosine annealing
    T_0 = 1
    T_mult = 2
    step_size = 2  # For step lr

    # saving models
    save_filename_G = "model.pth"
    save_filename_FFT = "FFT.pth"
    save_filename_D = "D.pth"

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_FFT = "FFT_latest.pth" 
    save_filename_latest_D = "D_latest.pth"

    log_interval = 100  # the number of iterations (default: 10) to print at
    save_ckpt_interval = log_interval * 10
    save_copy_every_epochs = 10
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    # See models/get_model.py for registry
    # model = "unet-128-pixelshuffle-invert"
    pixelshuffle_ratio = 2
    grad_lambda = 0.0
 
    G_finetune_layers = []  # None implies all

    num_groups = 8  # Group norm

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_adversarial = 0.6
    lambda_contextual = 0.0
    lambda_perception = 1.2  # 0.006
    lambda_image = 1  # mse
    lambda_l1 = 0 # l1

    resume = False
    finetune = False  # Wont load loss or epochs
    concat_input = False
    zero_conv = False
    # ---------------------------------------------------------------------------- #
    # Inference Args
    # ---------------------------------------------------------------------------- #
    inference_mode = "latest"
    assert inference_mode in ["latest", "best"]

    # ---------------------------------------------------------------------------- #
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distdataparallel = False
    val_train = False
    static_val_image = ""




def ours_diffusercam_mulnew_unet_padding_decode_sim():
    exp_name = "fft-mulnew9-diffusercam_unet_padding_decode_sim"
    batch_size = 5
    num_threads = 5
    lambda_adversarial = 0.0
    multi = 9
    use_spatial_weight = True
    lambda_perception = 0.05
    preprocess_with_unet = True
    psf_height = 270 * 2
    psf_width = 480 * 2
    decode_sim = True
  

def infer_train():
    val_train = True


named_config_ll = [
    ours_diffusercam_mulnew_unet_padding_decode_sim,
    infer_train
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_config_ll:
        ex.named_config(named_config)
    return ex

fft_args = SimpleNamespace(**fft_args_dict)

if __name__ == "__main__":
    str_named_config_ll = [str(named_config) for named_config in named_config_ll]
    print("\n".join(str_named_config_ll))
