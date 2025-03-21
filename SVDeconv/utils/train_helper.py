"""
Train Script for Unet

@MOD: 8th April 2019

@py3.6+

@requirements: See utils/requirments.txt
"""
# Libraries
from utils.model_serialization import load_state_dict

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.adamw import AdamW


# Torch Libs
import torch
import torch.distributed as dist

import logging

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *


def reduce_loss_dict(loss_dict, world_size):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    if world_size < 2:
        return {k: v.item() for k, v in loss_dict.items()}
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def pprint_args(args):
    """
    Pretty print args
    """
    string = str(args)
    string_ll = string.replace("Tupperware(", "").rstrip(")").split(", ")
    string_ll = sorted(string_ll, key=lambda x: x.split("=")[0].lower())

    string_ll = [
        f"*{line.split('=')[0]}* = {line.split('=')[-1]}" for line in string_ll
    ]
    string = "\n".join(string_ll)

    return string


def get_optimisers(G, FFT, args):
    g_optimizer = torch.optim.AdamW(
        G.parameters(), lr=args.learning_rate, betas=(args.beta_1, args.beta_2)
    )
    fft_optimizer = torch.optim.AdamW(
        FFT.parameters(), lr=args.fft_learning_rate, betas=(args.beta_1, args.beta_2)
    )

    if args.lr_scheduler == "cosine":
        g_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            g_optimizer, T_0=args.T_0, T_mult=args.T_mult
        )
        fft_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            fft_optimizer, T_0=args.T_0, T_mult=args.T_mult
        )
    else:
        g_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            g_optimizer, step_size=args.step_size
        )
        fft_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            fft_optimizer, step_size=args.step_size
        )

    return (g_optimizer, fft_optimizer), (g_lr_scheduler, fft_lr_scheduler)


def load_models(G, FFT, g_optimizer, fft_optimizer, args, is_local_rank_0=True):
    # ...existing code...
    if args.resume:
        try:
            if args.inference_mode == "latest":
                ckpt_G = torch.load(
                    args.ckpt_dir / args.save_filename_latest_G, map_location=args.device
                )
                ckpt_FFT = torch.load(
                    args.ckpt_dir / args.save_filename_latest_FFT, map_location=args.device
                )
            else:
                ckpt_G = torch.load(
                    args.ckpt_dir / args.save_filename_G, map_location=args.device
                )
                ckpt_FFT = torch.load(
                    args.ckpt_dir / args.save_filename_FFT, map_location=args.device
                )

            G.load_state_dict(ckpt_G["model"])
            FFT.load_state_dict(ckpt_FFT["model"])

            if not args.finetune:
                g_optimizer.load_state_dict(ckpt_G["optimizer"])
                fft_optimizer.load_state_dict(ckpt_FFT["optimizer"])
                global_step = ckpt_G["global_step"]
                epoch = ckpt_G["epoch"]
                loss = ckpt_G["loss"]
            else:
                global_step = 0
                epoch = 0
                loss = float("inf")

            if is_local_rank_0:
                logging.info(f"Loaded checkpoint from epoch {epoch}")
        except:
            if is_local_rank_0:
                logging.info("Could not load checkpoint")
            global_step = 0
            epoch = 0
            loss = float("inf")
    else:
        global_step = 0
        epoch = 0
        loss = float("inf")

    return (G, FFT), (g_optimizer, fft_optimizer), global_step, epoch, loss


def save_weights(
    global_step: int,
    epoch: int,
    G: "nn.Module" = None,
    FFT: "nn.Module" = None,
    g_optimizer: "optim" = None,
    fft_optimizer: "optim" = None,
    loss: "float" = None,
    is_min: bool = True,
    args: "tupperware" = None,
    tag: str = "latest",
    is_local_rank_0: bool = True,
):
    if is_min or tag == "latest":
        if is_local_rank_0:
            logging.info(f"Epoch {epoch + 1} saving weights")
        if G:
            # Gen
            G_state = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "state_dict": G.state_dict(),
                "optimizer": g_optimizer.state_dict(),
                "loss": loss,
            }
            save_filename_G = (
                args.save_filename_latest_G if tag == "latest" else args.save_filename_G
            )

            path_G = str(args.ckpt_dir / save_filename_G)
            torch.save(G_state, path_G)

            # Specific saving
            if epoch % args.save_copy_every_epochs == 0 and tag == "latest":
                save_filename_G = f"Epoch_{epoch}_{save_filename_G}"

                path_G = str(args.ckpt_dir / save_filename_G)
                torch.save(G_state, path_G)

        if FFT:
            # for _ in range(args.multi):
            #     print("------psf diff:",torch.sum(torch.abs(FFT.module.wiener_crop[0] - FFT.module.wiener_crop[_])))
            # FFT
            FFT_state = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "state_dict": FFT.state_dict(),
                # "optimizer": fft_optimizer.state_dict(),
                "loss": loss,
            }
            save_filename_FFT = (
                args.save_filename_latest_FFT
                if tag == "latest"
                else args.save_filename_FFT
            )

            path_FFT = str(args.ckpt_dir / save_filename_FFT)
            torch.save(FFT_state, path_FFT)

            if epoch % args.save_copy_every_epochs == 0 and tag == "latest":
                save_filename_FFT = f"Epoch_{epoch}_{save_filename_FFT}"

                path_FFT = str(args.ckpt_dir / save_filename_FFT)
                torch.save(FFT_state, path_FFT)

        
    else:
        if is_local_rank_0:
            logging.info(f"Epoch {epoch + 1} NOT saving weights")


class SmoothenValue(object):
    "Create a smooth moving average for a value (loss, etc) using `beta`."

    def __init__(self, beta: float = 0.9):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


# Dictionary based loss collectors
# See train.py for usage
class AvgLoss_with_dict(object):

    """
    Utility class for storing running averages of losses
    """

    def __init__(self, loss_dict: "Dict", args: "tupperware", count: int = 0):
        self.args = args
        self.count = count
        self.loss_dict = loss_dict

    def reset(self):
        self.count = 0
        for k in self.loss_dict:
            self.loss_dict[k] = 0.0

    def __add__(self, loss_dict: "Dict"):
        self.count += 1

        assert loss_dict.keys() == self.loss_dict.keys(), "Keys donot match"

        for k in self.loss_dict:
            self.loss_dict[k] += (loss_dict[k] - self.loss_dict[k]) / self.count

        return self


class ExpLoss_with_dict(object):
    def __init__(self, loss_dict: "Dict", args: "tupperware"):
        """
        :param dict: Expects default dict
        """
        self.args = args
        self.loss_dict = loss_dict
        self.set_collector()

    def set_collector(self):
        self.collector_dict = {}
        for k in self.loss_dict:
            self.collector_dict[k + "_collector"] = SmoothenValue()

    def __add__(self, loss_dict: "Dict"):
        assert loss_dict.keys() == self.loss_dict.keys(), "Keys donot match"
        for k in self.loss_dict:
            self.collector_dict[k + "_collector"].add_value(loss_dict[k])
            self.loss_dict[k] = self.collector_dict[k + "_collector"].smooth

        return self
