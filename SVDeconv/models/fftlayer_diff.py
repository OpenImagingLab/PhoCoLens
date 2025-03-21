import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from sacred import Experiment
import numpy as np
import torch.nn as nn
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)

from config import initialise
from utils.ops import roll_n
from utils.tupperware import tupperware
import cv2
if TYPE_CHECKING:
    from utils.typing_alias import *

from PIL import Image

ex = Experiment("FFT-Layer")
ex = initialise(ex)

SIZE = 270, 480

def transform(image, gray=False):
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    # average the RGB channels
    # image = image.mean(0, keepdim=True)
    return image

def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)

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
        for dim in range(1,3):
            psf = roll_n(psf, axis=dim, n=psf.shape[dim] // 2)


    # Perform 2D FFT
    H = torch.fft.fft2(psf)

    # Compute the absolute square of H
    H_conj = torch.conj(H)
    Habsq = H * H_conj
    # print("Habsq.real , Gamma", Habsq.real, Gamma)
    # Create Wiener filter
    W = torch.conj(H) / (Habsq.real + Gamma)

    # Perform 2D inverse FFT
    wiener_mat = torch.fft.ifft2(W)

    # Extract the real part
    return wiener_mat.real


class FFTLayer_diff(nn.Module):
    def __init__(self, args: "tupperware"):
        super().__init__()
        self.args = args
        requires_grad = True
        psf = load_psf(args.psf_mat)
        self.psf = psf
        if len(psf.shape) == 2:
            psf = psf.unsqueeze(0)


        psf_crop_top = args.psf_centre_x - args.psf_crop_size_x // 2
        psf_crop_bottom = args.psf_centre_x + args.psf_crop_size_x // 2
        psf_crop_left = args.psf_centre_y - args.psf_crop_size_y // 2
        psf_crop_right = args.psf_centre_y + args.psf_crop_size_y // 2

        psf_crop = psf[:, psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]

        _, self.psf_height, self.psf_width = psf_crop.shape
        # print("psf_crop.shape", psf_crop.shape)
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

        for dim in range(1,3):
            self.fft_layer = roll_n(
                self.fft_layer, axis=dim, n=self.fft_layer.size(dim) // 2
            )

        # Make 1 x 1 x H x W
        self.fft_layer = self.fft_layer.unsqueeze(0)

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
        # Pad to psf_height, psf_width
        img = F.pad(
            img, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2)
        )
        # Use mask
        if self.args.use_mask:
            img = img * self.mask

        # Do FFT convolve
        img_deconv = fft_conv2d(img, self.fft_layer) * self.normalizer
        # Centre Crop
        img_deconv = img_deconv[
            :,
            :,
            fft_h // 2 - img_h // 2 : fft_h // 2 + img_h // 2,
            fft_w // 2 - img_w // 2 : fft_w // 2 + img_w // 2,
        ]
        return img_deconv


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    model = FFTLayer_diff(args).to(args.device)
    img = torch.rand(1, 4, 1280, 1408).to(args.device)

    model(img)
