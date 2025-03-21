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
# from waveprop.devices import  SensorParam
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.fftlayer import FFTLayer
from config import fft_args
FFT = FFTLayer(fft_args)
sensor = dict(size = np.array([4.8e-6 * 1518, 4.8e-6 * 2012]))


class DirectSensorFarFieldSimulator(FarFieldSimulator):
    """
    Subclass of FarFieldSimulator that allows direct initialization of the sensor without relying on sensor_dict.
    The sensor parameter should be a dictionary containing necessary sensor parameters, such as SensorParam.SIZE.
    """

    def __init__(
        self,
        object_height,
        scene2mask,
        mask2sensor,
        sensor,  # Directly pass the sensor dictionary here
        psf=None,
        output_dim=None,
        snr_db=None,
        max_val=255,
        device_conv="cpu",
        random_shift=False,
        is_torch=False,
        quantize=True,
        return_float=True,
        **kwargs
    ):
        # Handle axes and output dtype based on tensor type
        if is_torch:
            self.axes = (-2, -1)
            output_dtype = torch.float32
            if quantize and not return_float:
                output_dtype = torch.uint8
        else:
            self.axes = (0, 1)
            output_dtype = np.float32
            if quantize and not return_float:
                output_dtype = np.uint8
        self.is_torch = is_torch

        # Initialize resizing parameters
        self.object_height = object_height
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.sensor = sensor  # Direct assignment of the sensor dictionary
        self.random_shift = random_shift
        self.quantize = quantize

        # Handle PSF and convolution setup
        if psf is not None:
            self.device_conv = device_conv
            self.set_psf(psf)

            # Sensor output parameters
            self.output_dim = output_dim
            self.snr_db = snr_db
            self.max_val = max_val
            self.output_dtype = output_dtype
        else:
            self.fft_shape = None
            assert output_dim is not None, "output_dim must be specified when PSF is not provided"
            self.conv_dim = np.array(output_dim)


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate the lensless capture')
    parser.add_argument('--psf_path', default="data/phase_psf/psf.npy", help='psf folder path')
    parser.add_argument('--obj_path', default=None, help='object folder path')
    parser.add_argument('--save_path', default="output/decode_and_sim_rgb", help='save folder path')

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


def load_sim_save(simulator, obj_path, save_path):
    # load object
    obj = cv2.imread(obj_path)
    # obj = cv2.normalize(obj, None, 0, 255, cv2.NORM_MINMAX)

    # simulate
    obj = torch.tensor(obj).permute(2, 0, 1).unsqueeze(0).float()
    img = simulator.propagate(obj)
    #normalize img
  
    # img = img - np.min(img)
    # img = img / np.max(img) * 255

    # capture = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    
    capture = img
    # decoded = capture
    decoded = FFT(capture)
    decoded = (decoded - decoded.min()) / (decoded.max() - decoded.min())
    decoded = decoded * 255
    decoded = decoded.squeeze().permute(1, 2, 0).detach().numpy().astype(np.uint8)    


    save_path = os.path.join(save_path, os.path.basename(obj_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"save_path: {save_path}")
    cv2.imwrite(save_path, decoded)
   

if __name__ == "__main__":
    args = parse_args()
    psf_path = args.psf_path
    obj_path = args.obj_path
    save_path = args.save_path

    # load psf
    psf = np.load(psf_path)
    # add last dimension
    psf = psf[..., None]
    # print(psf.shape)
    psf = crop_and_padding(psf)
    psf = torch.tensor(psf).permute(2, 0, 1).unsqueeze(0).float()
    # transfer the psf 
    simulator = DirectSensorFarFieldSimulator(object_height = 0.4, scene2mask = 0.434, mask2sensor = 2e-3, sensor = sensor, psf = psf, is_torch=True, quantize=False, return_float=True)
    print("obj_path: ", obj_path, os.path.isdir(obj_path))
    if os.path.isdir(obj_path):
        obj_path_list = os.listdir(obj_path)
        obj_path_list = [os.path.join(obj_path, obj_path_i) for obj_path_i in obj_path_list]
        for obj_path_i in obj_path_list:
            load_sim_save(simulator, obj_path_i, save_path)
    else:
        load_sim_save(simulator, obj_path, save_path)

# python tools/decode_and_sim_rgb.py --psf_path data/phase_psf/psf.npy --obj_path /root/caixin/StableSR/data/flatnet_val/gts --save_path /root/caixin/StableSR/data/flatnet_val/sim_captures
# python tools/decode_and_sim_rgb.py --obj_path  data/flatnet/inputs/n01440764_457.png

# python tools/decode_and_sim_rgb.py  --obj_path /root/StableSR/data/flatnet_val/gts --save_path /root/StableSR/data/flatnet_val/sim_captures

# python tools/decode_and_sim_rgb.py  --obj_path /root/RawSense/LenslessPiCam/outputs/2024-08-12/11-43-17/SimPhlatCam_raw_1518x2012 --save_path /root/StableSR/data/flatnet_val/decoded_sim_captures_disfa


#python tools/decode_and_sim_rgb.py  --obj_path /root/StableSR/data/flatnet_sim_output_384_val/inputs --save_path /root/StableSR/data/flatnet_sim_output_384_val/decoded_sim_captures

#python tools/decode_and_sim_rgb.py --psf_path data/phase_psf/psf.npy --obj_path data/flatnet_val/gts --save_path data/flatnet_val/decoded_sim_captures