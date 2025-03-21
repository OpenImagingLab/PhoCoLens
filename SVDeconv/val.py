"""
Val Script for Phase/Amp mask
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import logging
import cv2
from pathlib import Path
# Torch Libs
import torch
from torch.nn import functional as F
import time
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from utils.tupperware import tupperware
from models import get_model
from metrics import PSNR
from config import initialise
from skimage.metrics import structural_similarity as ssim
from utils.model_serialization import load_state_dict

# LPIPS
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.ops import rggb_2_rgb, unpixel_shuffle
from utils.train_helper import load_models, AvgLoss_with_dict

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


@ex.config
def config():
    gain = 1.0
    tag = "384"


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    args.batch_size = 1

    # Set device, init dirs
    device = args.device
    # dir_init(args)

    # ADMM or not
    interm_name = "fft" 


    # Get data
    data = get_dataloaders(args)

    # Model
    G, FFT = get_model.model(args)

    ckpt_dir = Path("ckpts_phlatcam") / args.exp_name
    model_gen_path = ckpt_dir / "model_latest.pth"
    model_fft_path = ckpt_dir / "FFT_latest.pth"
    print("model_gen_path.exists()", model_gen_path.exists(), "model_fft_path.exists()", model_fft_path.exists())
    print("model_gen_path", model_gen_path, "model_fft_path", model_fft_path)
    if  model_gen_path.exists() and model_fft_path.exists():
        logging.info(f"Loading model from {model_gen_path}")
        gen_ckpt = torch.load(model_gen_path, map_location=torch.device("cpu"))
        fft_ckpt = torch.load(model_fft_path, map_location=torch.device("cpu"))

    #     # G.load_state_dict(gen_ckpt["state_dict"])
        load_state_dict(G, gen_ckpt["state_dict"])
        load_state_dict(FFT, fft_ckpt["state_dict"])

    G = G.to(device)
    FFT = FFT.to(device)

    # LPIPS Criterion
    lpips_criterion = loss_fn_alex.to(device)

    # Load Models
    (G, FFT), _, global_step, start_epoch, loss = load_models(
        G,
        FFT,
        g_optimizer=None,
        fft_optimizer=None,
        args=args,
    )

    _metrics_dict = {
        "PSNR": 0.0,
        "LPIPS_01": 0.0,
        "LPIPS_11": 0.0,
        "SSIM": 0.0,
        "Time": 0.0,
    }
    avg_metrics = AvgLoss_with_dict(loss_dict=_metrics_dict, args=args)

    if args.val_train:
        logging.info("Validating on train set.")
        data.val_loader = data.train_loader
        
    logging.info(
        f"Loaded experiment {args.exp_name}, dataset {args.dataset_name}, trained for {start_epoch} epochs."
    )
  
    # Run val for an epoch
    avg_metrics.reset()
    pbar = tqdm(range(len(data.val_loader) * args.batch_size), dynamic_ncols=True)

    if args.device == "cuda:0":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        start = end = 0

    # Val and test paths
    val_path = args.output_dir / "val" if not args.val_train else args.output_dir / "train"
    val_path.mkdir(exist_ok=True, parents=True)

 

    acc_time = 0.0
    with torch.no_grad():
        G.eval()
        FFT.eval()
        for i, batch in enumerate(data.val_loader):
            metrics_dict = defaultdict(float)

            source, target, filename = batch
            source, target = (source.to(device), target.to(device))

            if args.device == "cuda:0" and i:
                start.record()
            start_time = time.time()
            fft_output = FFT(source)
            
          

            # Unpixelshuffle
            fft_unpixel_shuffled = unpixel_shuffle(fft_output, args.pixelshuffle_ratio)
            output_unpixel_shuffled = G(fft_unpixel_shuffled)

            output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)
            
            acc_time += time.time() - start_time

            if args.device == "cuda:0" and i:
                end.record()
                torch.cuda.synchronize()
                metrics_dict["Time"] = start.elapsed_time(end)
            else:
                metrics_dict["Time"] = 0.0

            # PSNR
            metrics_dict["PSNR"] += PSNR(output, target).item()
          
            metrics_dict["LPIPS_01"] += lpips_criterion(
                output.mul(0.5).add(0.5), target.mul(0.5).add(0.5)
            ).mean().item()
            # print(filename, lpips_criterion(
            #     output.mul(0.5).add(0.5), target.mul(0.5).add(0.5)
            # ).mean().item())
            metrics_dict["LPIPS_11"] += lpips_criterion(output, target).mean().item()

            for e in range(args.batch_size):
                # Compute SSIM
                fft_output_vis = []
     
      
                in_c = fft_output[e].shape[0]
                for i in range(in_c // 4):
                    fft_output_vis.append(rggb_2_rgb(fft_output[e][4*i:4*i+4]).mul(0.5).add(0.5))

                for i in range(len(fft_output_vis)):
                    fft_output_vis[i] = (fft_output_vis[i] - fft_output_vis[i].min()) / (
                        fft_output_vis[i].max() - fft_output_vis[i].min()
                    )
                    fft_output_vis[i] = fft_output_vis[i].permute(1, 2, 0).cpu().detach().numpy()


                output_numpy = (
                    output[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                target_numpy = (
                    target[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                metrics_dict["SSIM"] += ssim(
                    target_numpy, output_numpy, multichannel=True, data_range=1.0,channel_axis = -1
                )

                # Dump to output folder
                name = filename[e].replace(".JPEG", ".png")
                parent = name.split("_")[0]
                path = val_path / parent
                path.mkdir(exist_ok=True, parents=True)
                path_output = path / ("output_" + name)
                path_fft = path / (f"{interm_name}_" + name)

                cv2.imwrite(
                    str(path_output), (output_numpy[:, :, ::-1] * 255.0).astype(int)
                )
                for i in range(len(fft_output_vis)):
                    cv2.imwrite(
                        str(path_fft).replace(".png", f"_{i}.png"), (fft_output_vis[i][:, :, ::-1] * 255.0).astype(int)
                    )
              
            metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
            avg_metrics += metrics_dict

            pbar.update(args.batch_size)
            pbar.set_description(
                f"Val Epoch : {start_epoch} Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f} | SSIM: {avg_metrics.loss_dict['SSIM']:.3f} | LPIPS_01: {avg_metrics.loss_dict['LPIPS_01']:.3f}| LPIPS_11: {avg_metrics.loss_dict['LPIPS_11']:.3f}"
            )

        with open(val_path / "metrics.txt", "w") as f:
            L = [
                f"exp_name:{args.exp_name} trained for {start_epoch} epochs\n",
                f"Inference mode {args.inference_mode}\n",
                "Metrics \n\n",
            ]
            L = L + [f"{k}:{v}\n" for k, v in avg_metrics.loss_dict.items()]
            f.writelines(L)
        # print("acc_time", acc_time)
        # if data.test_loader:
        #     pbar = tqdm(
        #         range(len(data.test_loader) * args.batch_size), dynamic_ncols=True
        #     )
        #     for i, batch in enumerate(data.test_loader):

        #         source, filename = batch
        #         source = source.to(device)

        #         fft_output = FFT(source)


        #         # Unpixelshuffle
        #         fft_unpixel_shuffled = unpixel_shuffle(
        #             fft_output, args.pixelshuffle_ratio
        #         )
        #         output_unpixel_shuffled = G(fft_unpixel_shuffled)

        #         output = F.pixel_shuffle(
        #             output_unpixel_shuffled, args.pixelshuffle_ratio
        #         )

        #         for e in range(args.batch_size):
        #             fft_output_vis = rggb_2_rgb(fft_output[e]).mul(0.5).add(0.5)
                

        #             fft_output_vis = (fft_output_vis - fft_output_vis.min()) / (
        #                 fft_output_vis.max() - fft_output_vis.min()
        #             )

        #             fft_output_vis = (
        #                 fft_output_vis.permute(1, 2, 0).cpu().detach().numpy()
        #             )

        #             output_numpy = (
        #                 output[e]
        #                 .mul(0.5)
        #                 .add(0.5)
        #                 .permute(1, 2, 0)
        #                 .cpu()
        #                 .detach()
        #                 .numpy()
        #             )
        #             # Dump to output folder
        #             # Phase and amplitude are nested
        #             name = filename[e].replace(".JPEG", ".png")
        #             parent, name = name.split("/")
        #             path = test_path / parent
        #             path.mkdir(exist_ok=True, parents=True)
        #             path_output = path / ("output_" + name)
        #             path_fft = path / (f"{interm_name}_" + name)
        #             cv2.imwrite(
        #                 str(path_output),
        #                 (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
        #             )
        #             cv2.imwrite(
        #                 str(path_fft),
        #                 (fft_output_vis[:, :, ::-1] * 255.0).astype(np.int),
        #             )

        #         pbar.update(args.batch_size)
        #         pbar.set_description(f"Test Epoch : {start_epoch} Step: {global_step}")
