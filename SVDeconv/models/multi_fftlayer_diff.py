import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from sacred import Experiment
import numpy as np
import torch.nn as nn
import copy
from utils.ops import unpixel_shuffle

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
from models.unet import UNet270480

ex = Experiment("FFT-Layer")
ex = initialise(ex)

SIZE = 270, 480

def transform(image, gray=False):
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    image = image.mean(0, keepdim=True)
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

def generate_vertices(K, H, W):
    step_h = H // (K ** 0.5 * 2) 
    step_w = W // (K ** 0.5 * 2)
    vertices = [(h, w) for h in torch.arange(step_h, H-step_h + 1, step=step_h * 2) for w in torch.arange(step_w, W-step_w + 1, step=step_w * 2)]
    # print(vertices)
    return torch.tensor(vertices[:K])  # 返回前K个均匀分布的点

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class SpatialVaryWeight(nn.Module):
    #add weight for each spatial position and each deconved image
    # input: deconved image, shape: (B, self.multi, C, H, W), C = 4 in our case
    # output: weight, shape: (B, C, H, W)
    def __init__(self, args: "tupperware"):
        super(SpatialVaryWeight, self).__init__()
        self.args = args
        self.multi = args.multi
        # Target image (eg: 384) dims
        img_h = self.args.image_height
        img_w = self.args.image_width
        weight_update = args.weight_update
        self.weight = nn.Parameter(torch.rand(self.multi, img_h, img_w), requires_grad=weight_update)
        self.init_weight()
    def forward(self, img):
        weight = F.softmax(self.weight, dim=0)
        weight = weight.unsqueeze(0).unsqueeze(2)
        weighted_img = img * weight
        result = weighted_img.sum(dim=1)

        # save_path = "weight.png"
        # cv2.imwrite("weight.png", (self.weight.data[1].cpu() * 255).numpy().astype(np.uint8))


        return result
    
    # initialize the weight with spatial varying deconvlution prior
    def init_weight(self):
        H = self.args.image_height
        W = self.args.image_width
        # generate the vertices
        vertices = generate_vertices(self.multi, H, W).float()
        # calculate the distance between each pixel and each vertex
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((y_grid, x_grid), dim=-1).float()  # Shape: (H, W, 2)
        vertices_expanded = vertices.unsqueeze(1).unsqueeze(1)
        grid_expanded = grid.unsqueeze(0)
        dists = torch.sqrt(((grid_expanded - vertices_expanded) ** 2).sum(dim=-1)) / (H + W)
        # print("dists.shape", dists)
        weights = 1 / (dists + 1e-6)
        normalized_weights = weights / weights.sum(dim=0, keepdim=True)
        self.weight.data = normalized_weights
        # visualize the weight
    

class MultiFFTLayer_diff(nn.Module):
    def __init__(self, args: "tupperware"):
        super().__init__()
        self.args = args
        # No grad if you're not training this layer
        requires_grad = not (args.fft_epochs == args.num_epochs)
        
        # requires_grad = args.fft_requires_grad
        # if args.psf_mat.endswith(".npy"):
        psf = load_psf(args.psf_mat)
        # elif args.psf_mat.endswith(".png") or args.psf_mat.endswith(".jpg"):
        #     psf = torch.tensor(cv2.imread(args.psf_mat, cv2.IMREAD_GRAYSCALE)).float()
        if len(psf.shape) == 2:
            psf = psf.unsqueeze(0)

        self.multi = args.multi + 1
        self.zero_conv = args.zero_conv
        self.preprocess_with_unet = args.preprocess_with_unet
        psf_crop_top = args.psf_centre_x - args.psf_crop_size_x // 2
        psf_crop_bottom = args.psf_centre_x + args.psf_crop_size_x // 2
        psf_crop_left = args.psf_centre_y - args.psf_crop_size_y // 2
        psf_crop_right = args.psf_centre_y + args.psf_crop_size_y // 2

        psf_crop = psf[:,psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]

        _, self.psf_height, self.psf_width = psf_crop.shape

        self.psf_crop = nn.Parameter(psf_crop.repeat(self.multi, 1, 1, 1), requires_grad=requires_grad)
        if self.zero_conv:
            self.psf_crop = nn.Parameter(psf_crop.unsqueeze(0), requires_grad=requires_grad)
            self.zero_res_conv = zero_module(nn.Conv2d(1, self.multi, 1))
        self.gamma = nn.Parameter(torch.tensor([args.fft_gamma] *  self.multi, dtype=torch.float32), requires_grad=requires_grad)
        unet_args = copy.deepcopy(args)
        unet_args.pixelshuffle_ratio = 2
        self.unet = UNet270480(unet_args, in_c=3)

        # wiener_crop = get_wiener_matrix(
        #     psf_crop, Gamma=args.fft_gamma, centre_roll=False
        # )
        # wiener_crop_tensor = wiener_crop.repeat(self.multi, 1, 1, 1)

        # self.wiener_crop =nn.Parameter(wiener_crop_tensor, requires_grad=requires_grad)

        # self.wiener_crop = nn.ParameterList(self.wiener_crop)
        self.normalizer = nn.Parameter(
            torch.tensor([1 / 0.0008]).reshape(1, 1, 1, 1).repeat(self.multi, 1, 1, 1), requires_grad=requires_grad
        )


        # if self.args.use_mask:
        #     mask = torch.tensor(np.load(args.mask_path)).float()
        #     self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, img):
        pad_x = self.args.psf_height - self.args.psf_crop_size_x
        pad_y = self.args.psf_width - self.args.psf_crop_size_y

        # Convert to 0...1
        # img = 0.5 * img + 0.5
        #center crop img to 1280x1408
        h, w = img.shape[2], img.shape[3]
        
        img = img[:, :, (h - self.psf_height)//2:(h + self.psf_height)//2, (w - self.psf_width)//2:(w + self.psf_width)//2]

        # Pad to psf_height, psf_width
        img = F.pad(
            img, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2), mode="replicate"
        )
        
        if self.preprocess_with_unet:
            # pixel shuffle
            img = unpixel_shuffle(img, 2)
            img = self.unet(img)
            img = F.pixel_shuffle(img, 2)

        if self.zero_conv:
            zero_res =  self.zero_res_conv(self.psf_crop).view(self.multi, 1, self.psf_height, self.psf_width)
            psf_crop =self.psf_crop + zero_res
            # print(psf_crop.shape)
        else:
            psf_crop = self.psf_crop
    
        self.fft_layers = []
        
        # Pad to psf_height, psf_width
        for i in range(self.multi):
            self.fft_layer = 1 * get_wiener_matrix(
            psf_crop[i], Gamma=self.gamma[i], centre_roll=False
        )
            self.fft_layer = F.pad(
            self.fft_layer, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2)
            )

            # Centre roll
            for dim in range(1,3):
                self.fft_layer = roll_n(
                    self.fft_layer, axis=dim, n=self.fft_layer.size(dim) // 2
                )

            # Make 1 x 3 x H x W
            self.fft_layer = self.fft_layer.unsqueeze(0)
            # print("self.fft_layer.shape", self.fft_layer.shape)
            # FFT Layer dims
            _, _, fft_h, fft_w = self.fft_layer.shape
            self.fft_layers.append(self.fft_layer)

        # Target image (eg: 384) dims
        img_h = self.args.image_height
        img_w = self.args.image_width

     

        imgs = []
        # Do FFT convolve
        for i in range(self.multi):
            img_ = fft_conv2d(img, self.fft_layers[i]) * self.normalizer[i]
            # Centre Crop
            img_ = img_[
                :,
                :,
                fft_h // 2 - img_h // 2 : fft_h // 2 + img_h // 2,
                fft_w // 2 - img_w // 2 : fft_w // 2 + img_w // 2,
            ]
            imgs.append(img_)
    
        if self.args.use_spatial_weight:
            img_0 = imgs[0]
            spatial_weight = SpatialVaryWeight(self.args).to(img.device)
            img_1 = spatial_weight(torch.stack(imgs[1:], dim=1))
            img_deconv = torch.cat([img_0, img_1], dim=1)
        else:
            img_deconv = torch.cat(imgs, dim=1)
        

        return img_deconv
      


@ex.automain
def main(_run):
    args = tupperware(_run.config)

    model = MultiFFTLayer(args).to(args.device)
    img = torch.rand(1, 4, 1280, 1408).to(args.device)

    model(img)
