from waveprop.simulation import FarFieldSimulator
# output the lensless capture with a given psf and a given object, and save the image
# Usage: python sim_capture.py --psf_path psf.png --obj_path obj.png 
# save the images in the visual folder output/sim_capture/psf_folder/obj_folder
import argparse
import os 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)
from PIL import Image
from waveprop.devices import  SensorParam
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fftlayer_diff_original import FFTLayer_diff

height = 270 * 4
width = 480 * 4
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

FFT = FFTLayer_diff(fft_args_dict)
PADD_SIZE = 270*3, 480*3
SIZE = 270, 480
sensor = dict(size = np.array([4.8e-6 * 1080, 4.8e-6 * 1920]))
# 4.8 * 10e-6 * 4 * 480 = 0.009216
# 2e-3 / 0.009216 * 0.4 = 0.08680555555555556

def transform(image, gray=False):
    # image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    # center padding
    image = F.pad(
        image,
        (PADD_SIZE[1] // 2, PADD_SIZE[1] // 2, PADD_SIZE[0] // 2, PADD_SIZE[0] // 2),
        mode='constant',
        value=0,
    )
    # average the RGB channels
    image = image.mean(0, keepdim=True)

    return image

def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate the lensless capture')
    parser.add_argument('--psf_path', default="data/phase_psf/psf.npy", help='psf folder path')
    parser.add_argument('--obj_path', default=None, help='object folder path')
    parser.add_argument('--save_path', default="output/decode_and_sim_rgb", help='save folder path')
    parser.add_argument('--adj', help='whether to adjust the light intensity', action='store_true',default=False)

    args = parser.parse_args()
    return args
# 384 * 4.8 * 10e-6 = 0.0018432
# 2e-3 / 0.0018432 * 0.4 = 0.434

def crop_and_padding(img, meas_crop_size_x=1280, meas_crop_size_y=1408, meas_centre_x=808, meas_centre_y=965, psf_height=1518, psf_width=2012, pad_meas_mode="replicate"):
    # crop
    img = torch.tensor(img)
    if meas_crop_size_x and meas_crop_size_y:
        crop_x = meas_centre_x - meas_crop_size_x // 2
        crop_y = meas_centre_y - meas_crop_size_y // 2

        # Replicate padding
        img = img[
            crop_x: crop_x + meas_crop_size_x,
            crop_y: crop_y + meas_crop_size_y,
            ]

        pad_x = psf_height - meas_crop_size_x
        pad_y = psf_width - meas_crop_size_y
        
        img = F.pad(
            img.permute(2, 0, 1).unsqueeze(0),
            (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2),
            mode=pad_meas_mode,
        )

        img = img.squeeze(0).permute(1, 2, 0)
        # resize for half
    img = img.numpy()
    # img = cv2.resize(img.numpy(), (meas_crop_size_y // 2, meas_crop_size_x // 2))
    
    # img = img[..., None]
    return img
# adjust the light intensity of img (add weight 0.6-1 on the input img), the center of the img is the brightest

    


def load_sim_save(simulator, obj_path, save_path, use_adjust_light_intensity=False):
    # load object
    # obj = cv2.imread(obj_path)
    obj = np.load(obj_path)
    print("obj shape: ", obj.shape)
    # obj = cv2.normalize(obj, None, 0, 255, cv2.NORM_MINMAX)


    # simulate
    obj = torch.tensor(obj).permute(2, 0, 1).unsqueeze(0).float()
    img = simulator.propagate(obj)

    capture = img
    # decoded = capture
    decoded = FFT(capture)
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())

    decoded = decoded.squeeze().permute(1, 2, 0).detach().numpy()

    # save_npy_path = os.path.join(save_path, os.path.basename(obj_path))
    # os.makedirs(os.path.dirname(save_npy_path), exist_ok=True)
    # print(f"save_path: {save_npy_path}")
  
    # np.save(save_npy_path, img)

    decoded = (decoded * 255).astype(np.uint8)    

    save_png_path = os.path.join(save_path + "_png", os.path.basename(obj_path))
    save_png_path = save_png_path.replace(".npy", ".png")
    os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    print(f"save_path: {save_png_path}")
    cv2.imwrite(save_png_path, decoded)
   

if __name__ == "__main__":
    args = parse_args()
    psf_path = args.psf_path
    obj_path = args.obj_path
    save_path = args.save_path
    adj = args.adj
    psf = load_psf(args.psf_path)
    # load psf
    # psf = np.load(psf_path)
    # # add last dimension
    # psf = psf[..., None]
    # print(psf.shape)
    # psf = crop_and_padding(psf)
    
    # transfer the psf 
    
    simulator = FarFieldSimulator(object_height = 0.4, scene2mask = 0.0868 * 4, mask2sensor = 2e-3, sensor = sensor, psf = psf, is_torch=True, quantize=False, return_float=True)

    if os.path.isdir(obj_path):
        obj_path_list = os.listdir(obj_path)
        obj_path_list = [os.path.join(obj_path, obj_path_i) for obj_path_i in obj_path_list]
        for obj_path_i in obj_path_list:
            if obj_path_i.endswith(".npy"):
                load_sim_save(simulator, obj_path_i, save_path, use_adjust_light_intensity=adj)
    else:
        load_sim_save(simulator, obj_path, save_path, use_adjust_light_intensity=adj)

# python tools/decode_and_sim_rgb.py --psf_path data/phase_psf/psf.npy --obj_path /root/caixin/StableSR/data/flatnet_val/gts --save_path /root/caixin/StableSR/data/flatnet_val/sim_captures
# python tools/decode_and_sim_rgb.py --obj_path  /root/caixin/StableSR/data/flatnet/gts/n01440764_457.png

# python tools/decode_and_sim_rgb_diff.py --psf_path /root/caixin/flatnet/data/diffusercam/psf.tiff --obj_path /root/caixin/flatnet/data/diffusercam/ground_truth_lensed_png/im_137.png --save_path /root/caixin/flatnet/data/diffusercam/sim_captures/im_137.png