"""
Convention

ours/naive-fft-(fft_h-fft_w)-learn-(learn_h-learn_w)-meas-(meas_h-meas-w)-kwargs

* Phlatcam: 1518 x 2012 (post demosiacking)
"""
from pathlib import Path
import torch
from types import SimpleNamespace

# Define FFT arguments once at the module level
fft_args_dict = {
    "psf_mat": Path("data/phlatcam/phase_psf/psf.npy"),
    "psf_height": 1518,
    "psf_width": 2012,
    "psf_centre_x": 808,
    "psf_centre_y": 965,
    "psf_crop_size_x": 1280,
    "psf_crop_size_y": 1408,
    "meas_height": 1518,
    "meas_width": 2012,
    "meas_centre_x": 808,
    "meas_centre_y": 965,
    "meas_crop_size_x": 1280,
    "meas_crop_size_y": 1408,
    "pad_meas_mode": "replicate",
    "image_height": 384,
    "image_width": 384,
    "fft_gamma": 2e4,  # Gamma for Weiner init
    "fft_requires_grad": False,
    "fft_epochs": 0,
    "use_mask": False,
}

def base_config():
    exp_name = "ours-fft-1280-1408-learn-1280-1408-meas-1280-1408"
    is_naive = "naive" in exp_name
    multi = 1
    use_spatial_weight = False
    weight_update = True

    # Use FFT arguments from the global definition
    locals().update(fft_args_dict)
    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data/phlatcam")
    output_dir = Path("output/phlatcam") / exp_name
    ckpt_dir = Path("ckpts/phlatcam") / exp_name
    run_dir = Path("runs/phlatcam") / exp_name  # Tensorboard

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    text_file_dir = image_dir / "text_files"
    train_source_list = text_file_dir / "train_source_imagenet_384_384_Feb_19.txt"
    train_target_list = text_file_dir / "train_target.txt"

    val_source_list = text_file_dir / "val_source_imagenet_384_384_Feb_19.txt"
    val_target_list = text_file_dir / "val_target.txt"

    test_skip_existing = True
    test_apply_gain = True

    dataset_name = "phlatcam" 

    shuffle = True
    train_gaussian_noise = 5e-3

    batch_size = 10
    num_threads = batch_size >> 1  # parallel workers

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    num_epochs = 100
    fft_epochs = num_epochs if is_naive else 0

    learning_rate = 3e-4
    fft_learning_rate = 4e-10

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

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_FFT = "FFT_latest.pth"

    log_interval = 100  # the number of iterations (default: 10) to print at
    save_ckpt_interval = log_interval * 10
    save_copy_every_epochs = 10
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    # See models/get_model.py for registry
    model = "unet-128-pixelshuffle-invert"
    pixelshuffle_ratio = 2
    grad_lambda = 0.0
  
    G_finetune_layers = []  # None implies all

    num_groups = 8  # Group norm

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_contextual = 0.0
    lambda_perception = 1.2  # 0.006
    lambda_image = 1  # mse
    lambda_l1 = 0 # l1

    resume = False
    finetune = False  # Wont load loss or epochs
    load_raw = False

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
    
    static_val_image = "n02165456_10030.png"


def ours_meas_1280_1408_svd():
    exp_name = "fft-svd-1280-1408-learn-1280-1408-meas-1280-1408"
    batch_size = 5
    num_threads = 5
    multi = 9
    use_spatial_weight = True
    load_raw = True


def ours_meas_1280_1408_decoded_sim_svd():
    exp_name = "fft-svd-1280-1408-meas-decoded_sim_spatial_weight"
    train_target_list =  "data/phlatcam/text_files/decoded_sim_captures_train.txt"
    val_target_list = "data/phlatcam/text_files/decoded_sim_captures_val.txt"
    batch_size = 5
    num_threads = 5
    use_spatial_weight = True
    multi = 9
    load_raw = True

def load_ckpt():
    exp_name = "fft-svd-1280-1408-learn-1280-1408-meas-decoded_sim_spatial_weight"

# def ours_meas_1280_1408_decoded_sim():
#     exp_name = "fft-svd-1280-1408-meas-decoded_sim_spatial_weight"
#     train_target_list =  "data/text_files/decoded_sim_captures_train.txt"
#     val_target_list = "data/text_files/decoded_sim_captures_val.txt"
#     batch_size = 5
#     num_threads = 5
#     use_spatial_weight = True
#     multi = 9
#     load_raw = True

def infer_train():
    val_train = True


named_config_ll = [
    ours_meas_1280_1408_svd,
    ours_meas_1280_1408_decoded_sim_svd,
    infer_train,
    load_ckpt
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
