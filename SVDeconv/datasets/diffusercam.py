from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
from sacred import Experiment

# Torch modules
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.distributed as dist
import os
import cv2
import numpy as np
from config import initialise
from pathlib import Path

if TYPE_CHECKING:
    from utils.typing_alias import *


ex = Experiment("data")
ex = initialise(ex)


SIZE = 270, 480

def region_of_interest(x):
    return x[..., 60:270, 60:440]


def transform(image, gray=False):
    # print(image.shape)
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    image = (image - 0.5) * 2
    return image


def sort_key(x):
    return int(x[2:-4])


def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)


class LenslessLearning(Dataset):
    def __init__(self, diffuser_images, ground_truth_images):
        """
        Everything is upside-down, and the colors are BGR...
        """
        self.xs = diffuser_images
        self.ys = ground_truth_images

    def read_image(self, filename):
        image = np.load(filename)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        diffused = self.xs[idx]
        ground_truth = self.ys[idx]
        # print(diffused, ground_truth)
        # print("hello!", np.load(diffused).shape, np.load(ground_truth).shape)
        
        x = transform(np.load(diffused))
        if ground_truth.name.endswith('.png'):
            y = np.array(Image.open(ground_truth))
            y = transform(y)
        else:
            y = transform(np.load(ground_truth))
        
        return x, y, str(diffused.name)


class LenslessLearningInTheWild(Dataset):
    def __init__(self, path):
        xs = []
        manifest = sorted((x.name for x in path.glob('*.npy')))
        for filename in manifest:
            xs.append(path / filename)

        self.xs = xs

    def read_image(self, filename):
        image = np.load(filename)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        diffused = self.xs[idx]
        x = transform(np.load(diffused))
        return x


class LenslessLearningCollection:
    def __init__(self, args):
        path = Path(args.image_dir)

        self.psf = load_psf(path / 'psf.tiff')

        train_diffused, train_ground_truth = load_manifest(path, 'dataset_train.csv', decode_sim = args.decode_sim)
        val_diffused, val_ground_truth = load_manifest(path, 'dataset_test.csv', decode_sim = args.decode_sim)

        self.train_dataset = LenslessLearning(train_diffused, train_ground_truth)
        self.val_dataset = LenslessLearning(val_diffused, val_ground_truth)
        self.region_of_interest = region_of_interest


def load_manifest(path, csv_filename, decode_sim = False):
    with open(path / csv_filename) as f:
        manifest = f.read().split()

    xs, ys = [], []
    for filename in manifest:
        x = path / 'diffuser_images' / filename.replace(".jpg.tiff", ".npy")
        if decode_sim:
            y = path / 'decode_sim_padding_png' / filename.replace(".jpg.tiff", ".png")
        else:
            y = path / 'ground_truth_lensed' / filename.replace(".jpg.tiff", ".npy")
        # if x.exists() and y.exists():
        #     print(f"Found {x} and {y}")
        xs.append(x)
        ys.append(y)
        # else:
        #     print(f"No file named {x}")
    return xs, ys