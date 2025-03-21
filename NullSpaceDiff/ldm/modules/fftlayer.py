import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from sacred import Experiment
import numpy as np
import torch.nn as nn

from pathlib import Path
from types import SimpleNamespace


def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fft_conv2d(input, kernel):
    """
    Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain!
    :param input: shape (B, Cin, H, W)
    :param kernel: shape (Cout, Cin, H, W)
    :param bias: shape of (B, Cout, H, W)
    :return:
    """
    input = torch.fft.fft2(input)
    kernel = torch.fft.fft2(kernel)

    # Compute the multiplication
    # (a+bj)*(c+dj) = (ac-bd)+(ad+bc)j
    # real = input[..., 0] * kernel[..., 0] - input[..., 1] * kernel[..., 1]
    # im = input[..., 0] * kernel[..., 1] + input[..., 1] * kernel[..., 0]

    # Stack both channels and sum-reduce the input channels dimension
    out_complex = input * kernel
    out = torch.fft.ifft2(out_complex).real.squeeze(-1)
    return out


def get_wiener_matrix(psf, Gamma: int = 20000, centre_roll: bool = True):
    """
    Get Wiener filter matrix from PSF.
    :param psf: The point spread function.
    :param Gamma: The regularization parameter.
    :param centre_roll: Boolean to determine whether to roll the PSF to the center.
    :return: The Wiener filter matrix.
    """

    if centre_roll:
        for dim in range(2):
            psf = roll_n(psf, axis=dim, n=psf.shape[dim] // 2)

    psf = psf.unsqueeze(0)

    # Perform 2D FFT
    H = torch.fft.fft2(psf)

    # Compute the absolute square of H
    H_conj = torch.conj(H)
    Habsq = H * H_conj

    # Create Wiener filter
    W = torch.conj(H) / (Habsq.real + Gamma)

    # Perform 2D inverse FFT
    wiener_mat = torch.fft.ifft2(W)

    # Extract the real part
    return wiener_mat.real[0]


class FFTLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # No grad if you're not training this layer
        # requires_grad = not (args.fft_epochs == args.num_epochs)
        
        requires_grad = args.fft_requires_grad

        psf = torch.tensor(np.load(args.psf_mat)).float()

        psf_crop_top = args.psf_centre_x - args.psf_crop_size_x // 2
        psf_crop_bottom = args.psf_centre_x + args.psf_crop_size_x // 2
        psf_crop_left = args.psf_centre_y - args.psf_crop_size_y // 2
        psf_crop_right = args.psf_centre_y + args.psf_crop_size_y // 2

        psf_crop = psf[psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]

        self.psf_height, self.psf_width = psf_crop.shape

        wiener_crop = get_wiener_matrix(
            psf_crop, Gamma=args.fft_gamma, centre_roll=False
        )

        self.wiener_crop = nn.Parameter(wiener_crop, requires_grad=requires_grad)

        self.normalizer = nn.Parameter(
            torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1), requires_grad=requires_grad
        )

        if self.args.use_mask:
            mask = torch.tensor(np.load(args.mask_path)).float()
            self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, img):
        pad_x = self.args.psf_height - self.args.psf_crop_size_x
        pad_y = self.args.psf_width - self.args.psf_crop_size_y

        # Pad to psf_height, psf_width
        self.fft_layer = 1 * self.wiener_crop
        self.fft_layer = F.pad(
            self.fft_layer, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2)
        )

        # Centre roll
        for dim in range(2):
            self.fft_layer = roll_n(
                self.fft_layer, axis=dim, n=self.fft_layer.size(dim) // 2
            )

        # Make 1 x 1 x H x W
        self.fft_layer = self.fft_layer.unsqueeze(0).unsqueeze(0)
        # print("self.fft_layer.shape", self.fft_layer.shape)
        # FFT Layer dims
        _, _, fft_h, fft_w = self.fft_layer.shape

        # Target image (eg: 384) dims
        img_h = self.args.image_height
        img_w = self.args.image_width

        # Convert to 0...1
        img = 0.5 * img + 0.5
        #center crop img to 1280x1408
        h, w = img.shape[2], img.shape[3]
        
        img = img[:, :, (h - self.psf_height)//2:(h + self.psf_height)//2, (w - self.psf_width)//2:(w + self.psf_width)//2]
        # print("img.shape", img.shape)
        # Pad to psf_height, psf_width
        img = F.pad(
            img, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2)
        )
        # print("img.shape", img.shape)
        # Use mask
        if self.args.use_mask:
            img = img * self.mask

        # Do FFT convolve
        img = fft_conv2d(img, self.fft_layer) * self.normalizer

        # Centre Crop
        img = img[
            :,
            :,
            fft_h // 2 - img_h // 2 : fft_h // 2 + img_h // 2,
            fft_w // 2 - img_w // 2 : fft_w // 2 + img_w // 2,
        ]
        return img





fft_args = {
    "psf_mat": Path("/root/caixin/flatnet/data/phase_psf/psf.npy"),
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
    # Change meas_crop_size_{x,y} to crop sensor meas. This will assume your sensor is smaller than the
    # measurement size. True measurement size is 1280x1408x4. Anything smaller than this requires padding of the
    # cropped measurement and then multiplying this with gaussian filtered rectangular box. For simplicity use the arguments
    # already set. Currently we are using full measurement. 
    "image_height": 384,
    "image_width": 384,
    "fft_gamma": 20000,  # Gamma for Weiner init
    "use_mask": False,  # Use mask for cropped meas only
    "mask_path": Path("data/phase_psf/box_gaussian_1280_1408.npy"),
    # use Path("box_gaussian_1280_1408.npy") for controlled lighting
    # use Path("box_gaussian_1280_1408_big_mask.npy") for uncontrolled lighting
    "fft_requires_grad": False,
} 

fft_args = SimpleNamespace(**fft_args)
