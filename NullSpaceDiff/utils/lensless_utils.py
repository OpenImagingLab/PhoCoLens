from waveprop.simulation import FarFieldSimulator
import argparse
import os 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from waveprop.devices import  SensorParam
from torch import nn
sensor = dict(size = np.array([4.8e-6 * 1518, 4.8e-6 * 2012]))

class LenslessSimulator(nn.Module):
    def __init__(self, psf_path, object_height = 0.4, scene2mask = 0.434, mask2sensor = 2e-3, sensor = sensor):
        super(LenslessSimulator, self).__init__()
        psf = np.load(psf_path)
        psf = torch.from_numpy(psf).float()
        psf = psf.unsqueeze(0).unsqueeze(0)
        psf = self.crop_and_padding(psf)
        # psf = psf.unsqueeze(-1)
        # psf = psf[..., None]
        # print(psf.shape)
        self.simulator = FarFieldSimulator(object_height, scene2mask, mask2sensor, sensor, psf, is_torch = True, quantize = False)

    def crop_and_padding(self, img, 
    meas_crop_size_x=1280, meas_crop_size_y=1408, meas_centre_x=808, meas_centre_y=965, psf_height=1518, psf_width=2012, pad_meas_mode="replicate"):
        crop_x = meas_centre_x - meas_crop_size_x // 2
        crop_y = meas_centre_y - meas_crop_size_y // 2
        crop_x_end = crop_x + meas_crop_size_x
        crop_y_end = crop_y + meas_crop_size_y
        if crop_x < 0:
            crop_x = 0
        if crop_y < 0:
            crop_y = 0
        # img shape: (B, C, H, W)
        img = img[:, :, crop_x:crop_x_end, crop_y:crop_y_end]
        # pad to psf size
        pad_x = psf_height - meas_crop_size_x
        pad_y = psf_width - meas_crop_size_y
        if pad_x < 0 or pad_y < 0:
            raise ValueError("psf size should be larger than meas_crop_size")
        #resize to half size
        # img = F.interpolate(img, scale_factor=0.5, mode="bilinear")
        # print("pad_x: {}, pad_y: {}".format(pad_x, pad_y))
        # print("img shape: {}".format(img.shape))
        img = F.pad(img, (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2), mode=pad_meas_mode)
        return img


    def forward(self, x):
        # print("max: {}, min: {}".format(torch.max(x), torch.min(x)))
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        # x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        x = self.simulator.propagate(x)
        # x = x / 1000

        # print("after sim conv max: {}, min: {}".format(torch.max(x), torch.min(x)))
        # x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        # transfer to -1-1
        # x = x * 2 - 1
        # x = self.crop_and_padding(x)
        return x

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
    "mask_path": Path("/root/caixin/flatnet/data/phase_psf/box_gaussian_1280_1408.npy"),
    # use Path("box_gaussian_1280_1408.npy") for controlled lighting
    # use Path("box_gaussian_1280_1408_big_mask.npy") for uncontrolled lighting
    "fft_requires_grad": False,
} 

def load_real_capture_as_tensor(data_path, is_cuda = True):
    real_capture = cv2.imread(data_path).astype(np.float32) / 255
    real_capture = (real_capture - np.min(real_capture)) / (np.max(real_capture) - np.min(real_capture))
    real_capture = torch.tensor(real_capture).permute(2, 0, 1).float().unsqueeze(0)
    if is_cuda:
        real_capture = real_capture.cuda()