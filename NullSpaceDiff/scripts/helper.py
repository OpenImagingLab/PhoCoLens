
import sys 
# sys.path.append('/mnt/workspace/RawSense/deep-image-prior/')
from utils.lensless_utils import LenslessSimulator
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.fftlayer import FFTLayer, fft_args
import numpy as np
from PIL import Image
import torch
import PIL
from torchvision.utils import save_image, make_grid
import os

class GuidanceDetails:
    def __init__(self) -> None:
        # guidance have the following parameters (attributes):
        # guidance_gt, guidance_func, guidance_loss, guidance_lr, guidance_step, guidance_range
        # guidance_gt: the ground truth image for guidance
        # guidance_func: the function used for guidance
        # guidance_loss: the loss function used for guidance
        # guidance_weight: the weight for the guidance loss
        # guidance_step: the number of steps for guidance
        # guidance_range: the range of timesteps for guidance
        # guidance_flag: calculate the gradient of x_t or x_0 (0 for x_0, 1 for x_t, default is 0)
        self.guidance_gt = None
        self.guidance_func = None
        self.guidance_loss = None
        self.guidance_weight = 0.1
        self.guidance_step = 1
        self.guidance_range = [100,1000]
        self.guidance_flag = 0

class LenslessGuidance(GuidanceDetails):
    def __init__(self, opt = None) -> None:
        super().__init__()
        self.lensless_simulator = LenslessSimulator("/root/caixin/flatnet/data/phase_psf/psf.npy")
        self.guidance_loss = F.mse_loss
        self.guidance_weight = 0
        self.guidance_step = 1
        self.guidance_range = [0,100]
        # self.guidance_range = [100,1000]

        self.guidance_flag = 0
        self.guidance_func = lambda x: self.lensless_simulator(x)
        if opt is not None:
            self.guidance_weight = opt.guidance_weight
            self.guidance_step = opt.guidance_step
            self.guidance_range = [opt.guidance_range[0], opt.guidance_range[1]]
            self.guidance_flag = opt.guidance_flag
        
        

class OriginGuidance(GuidanceDetails):
    def __init__(self) -> None:
        super().__init__()
        self.guidance_func = lambda x: x
        self.guidance_loss = F.mse_loss
        self.guidance_weight = 0.1
        self.guidance_step = 1
        self.guidance_range = [100,1000]
        self.guidance_flag = 0

class DDNMGuidance(object):
    def __init__(self, opt = None) -> None:
        super().__init__()
        self.forward_func = lambda x: normalize_img(self.lensless_simulator(x))
        self.inverse_func = lambda x: normalize_img(self.lensless_deconv(x))
        # self.forward_func = lambda x: self.lensless_simulator(x)
        # self.inverse_func = lambda x: self.lensless_deconv(x)
        # self.guidance_weight = 0.1
        self.lensless_simulator = LenslessSimulator("/root/caixin/flatnet/data/phase_psf/psf.npy")
        self.lensless_deconv = FFTLayer(fft_args)
        self.y = None

    def guidance_weight(self, t):
        # consine annealing, t = 0, weight = 0, t = 1000, weight = 1 with torch
        # return 0.5 * (1 - np.cos(np.pi * t / 50))
        # return 0.5 * (1 - torch.cos(torch.pi * t / 1000))
        return 0.3
     

    def __call__(self, x0_t, t=None):
        # print("x0_t shape: ", x0_t.shape)
        # print("y shape: ", self.y.shape)
        self.lensless_deconv = self.lensless_deconv.to(x0_t.device)
        # inverse_y = self.inverse_func(self.y)
        inverse_y = self.y
        inverse_y = F.interpolate(inverse_y, size = x0_t.shape[2:], mode = "bilinear")
        inverse_x0_t = self.inverse_func(self.forward_func(x0_t))
        inverse_x0_t = F.interpolate(inverse_x0_t, size = x0_t.shape[2:], mode = "bilinear")
        delta = inverse_y  - inverse_x0_t
        # delta = inverse_y  - x0_t
        # resize delta to x0_t size
        # delta = F.interpolate(delta, size = x0_t.shape[2:], mode = "bilinear")

        #visualize x0_t, inverse_y, inverse_x0_t, delta
        #concatenate x0_t, inverse_y, inverse_x0_t, delta
        # concat_image = torch.cat([x0_t, inverse_y, inverse_x0_t, delta], dim = 0)
        # concat_image = make_grid(concat_image, nrow = x0_t.shape[0])
        # os.makedirs("visualization", exist_ok = True)
        # save_image(concat_image, "visualization/concat_image_%d.png"%t.data.item(), normalize = True)
        return delta

def normalize_img(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = 2 * img - 1
    return img

if __name__ == "__main__":
    test_img_path = "/root/caixin/StableSR/data/flatnet_val/gts/n01818515_3376.png"
    def load_img(path):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        # print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.
    img = load_img(test_img_path)[0]
    img = img.clamp(-1,1)
    ddnm_guidance = DDNMGuidance()
    y = ddnm_guidance.forward_func(img)
    # y = normalize_img(y)
    print("y shape: ", y.shape)
    print("y max: ", y.max())
    print("y min: ", y.min())
    x_hat = ddnm_guidance.inverse_func(y)
    # x_hat = normalize_img(x_hat)
    print("x_hat shape: ", x_hat.shape)
    print("x_hat max: ", x_hat.max())
    print("x_hat min: ", x_hat.min())
    x_hat = (x_hat + 1) / 2
    save_image(x_hat, "x_hat.png", normalize = True)
    



        
        