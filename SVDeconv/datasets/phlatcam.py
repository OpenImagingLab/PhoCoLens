from dataclasses import dataclass
import logging

# Torch modules
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import cv2
import numpy as np
from pathlib import Path


def resize(img, factor):
    num = int(-np.log2(factor))

    for i in range(num):
        dim_x = img.shape[0]
        dim_y = img.shape[1]
        pad_x = 1 if dim_x % 2 == 1 else 0
        pad_y = 1 if dim_y % 2 == 1 else 0
        img = 0.25 * (
            img[: dim_x - pad_x : 2, : dim_y - pad_y : 2]
            + img[1::2, : dim_y - pad_y : 2]
            + img[: dim_x - pad_x : 2, 1::2]
            + img[1::2, 1::2]
        )

    return img


def get_img_from_raw(raw, dataset_name: str = "phlatcam"):
    raw_h, raw_w = raw.shape

    if dataset_name == "phlatcam":
        img = np.zeros((raw_h // 2, raw_w // 2, 4))

        img[:, :, 0] = raw[0::2, 0::2]  # r
        img[:, :, 1] = raw[0::2, 1::2]  # gr
        img[:, :, 2] = raw[1::2, 0::2]  # gb
        img[:, :, 3] = raw[1::2, 1::2]  # b

        img = torch.tensor(img)

    elif dataset_name == "phlatcam_admm":
        img = np.zeros((raw_h // 2, raw_w // 2, 3))

        img[:, :, 0] = raw[0::2, 0::2]  # r
        img[:, :, 1] = 0.5 * (raw[0::2, 1::2] + raw[1::2, 0::2])  # g
        img[:, :, 2] = raw[1::2, 1::2]  # b
        img = torch.tensor(img)

    return img


class PhaseMaskDataset(Dataset):
    """
    Assume folders have images one level deep
    """

    def __init__(
        self,
        args,
        mode: str = "train",
        max_len: int = None,
        is_local_rank_0: bool = True,
    ):
        super().__init__()

        assert mode in ["train", "val", "test"], "Mode can be train or val"
        self.mode = mode
        self.args = args
        self.image_dir = args.image_dir
        self.max_len = max_len
        self.source_path = None
        self.source_paths, self.target_paths = self._load_dataset()
        self.load_raw = args.load_raw


        if is_local_rank_0:
            logging.info(f"{mode.capitalize()} Set | Image Dir: {self.image_dir}")

    def _glob_images(self, file_list):
        with open(file_list) as f:
            source_paths = f.readlines()

        paths = [self.image_dir / Path(path.strip("\n")) for path in source_paths]
        return paths

    def _img_load(self, img_path: "Path" = None, img_mode="source", raw=[]):
        assert img_path or len(raw), "need either path or raw image"
        assert img_mode in ["source", "target", "source_png"], "img_mode not supported"
        if img_mode == "target":
            img = cv2.imread(str(img_path))[:, :, ::-1] / 255.0

            img = cv2.resize(img, (self.args.image_width, self.args.image_height))
        
        elif img_mode == "source_png":
            img = cv2.imread(str(img_path))[:, :, ::-1] / 255.0

        elif img_mode == "source":
            if not len(raw):
                raw = cv2.imread(str(img_path), -1)
            try:
                raw = raw / 4096.0
            except:
                breakpoint()

            img = get_img_from_raw(raw, self.args.dataset_name)

            # Crop
            if self.args.meas_crop_size_x and self.args.meas_crop_size_y:
                crop_x = self.args.meas_centre_x - self.args.meas_crop_size_x // 2
                crop_y = self.args.meas_centre_y - self.args.meas_crop_size_y // 2

                # Replicate padding
                img = img[
                    crop_x : crop_x + self.args.meas_crop_size_x,
                    crop_y : crop_y + self.args.meas_crop_size_y,
                ]

                pad_x = self.args.psf_height - self.args.meas_crop_size_x
                pad_y = self.args.psf_width - self.args.meas_crop_size_y

                img = F.pad(
                    img.permute(2, 0, 1).unsqueeze(0),
                    (pad_y // 2, pad_y // 2, pad_x // 2, pad_x // 2),
                    mode=self.args.pad_meas_mode,
                )

                img = img.squeeze(0).permute(1, 2, 0)

            if self.args.test_apply_gain and self.mode == "test":
                img = img / img.max() * (400 / 4096.0)

            if self.args.dataset_name == "phlatcam_admm":
                img = resize(img, 0.25)

        img = (img - 0.5) * 2  # Change range from -1,...,1
        img = np.transpose(img, (2, 0, 1))

        return img

    def _load_dataset(self):
        if self.mode == "train":
            source_paths = self._glob_images(self.args.train_source_list)[
                : self.max_len
            ]
            target_paths = self._glob_images(self.args.train_target_list)[
                : self.max_len
            ]

        elif self.mode == "val":
            source_paths = self._glob_images(self.args.val_source_list)[: self.max_len]
            target_paths = self._glob_images(self.args.val_target_list)[: self.max_len]

        elif self.mode == "test":
            source_paths = list(self.image_dir.glob(self.args.test_glob_pattern))
            target_paths = None

        return source_paths, target_paths

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source_path = self.source_paths[index]
        if self.source_path == None and os.path.exists(source_path):
            self.source_path = source_path

        if not os.path.exists(source_path):
            print(source_path, " does not exist")
            if self.source_path == None:
                print("No source path")
            source_path = self.source_path

        if self.load_raw:
            source = self._img_load(source_path, img_mode="source").float()
            
        else:
            source = self._img_load(source_path, img_mode="source_png")
            source = torch.from_numpy(source.copy()).float()

        if self.mode == "test":
            return source.float(), f"{source_path.parent.name}/{source_path.name}"

        # if self.mode == "train":
        #     source = source + torch.normal(
        #         torch.zeros_like(source), self.args.train_gaussian_noise
        #     )

        target_path = self.target_paths[index]
        target = self._img_load(target_path, img_mode="target")
        target = torch.from_numpy(target.copy()).float()
        # print(source.shape, target.shape)
        return source, target, source_path.name
  
