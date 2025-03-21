"""
Dataloaders
"""

# Libs
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
from datasets.phlatcam import PhaseMaskDataset
from datasets.diffusercam import LenslessLearningCollection
if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("data")
ex = initialise(ex)
@dataclass
class Data:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader = None

def get_dataloaders(args, is_local_rank_0: bool = True):
    """
    Get dataloaders for train and val

    Returns:
    :data
    """
    if "phlatcam" in args.dataset_name:
        train_dataset = PhaseMaskDataset(
            args, mode="train", is_local_rank_0=is_local_rank_0
        )
        val_dataset = PhaseMaskDataset(args, mode="val", is_local_rank_0=is_local_rank_0)
        test_dataset = PhaseMaskDataset(args, mode="val", is_local_rank_0=is_local_rank_0)
    elif "diffusercam" in args.dataset_name:
        dataset = LenslessLearningCollection(args)
        train_dataset = dataset.train_dataset
        val_dataset = dataset.val_dataset
        test_dataset = dataset.val_dataset
        # print("here")
    if is_local_rank_0:
        logging.info(
            f"Dataset: {args.dataset_name} Len Train: {len(train_dataset)} Val: {len(val_dataset)}  Test: {len(test_dataset)}"
        )

    train_loader = None
    val_loader = None
    test_loader = None

    if len(train_dataset):
        if args.distdataparallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )
            shuffle = False

        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=True,
            # drop_last=True,
            sampler=train_sampler,
        )

    if len(val_dataset):
        if args.distdataparallel:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )
            shuffle = False

        else:
            val_sampler = None
            shuffle = False

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=True,
            # drop_last=True,
            sampler=val_sampler,
        )

    if len(test_dataset):
        if args.distdataparallel:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=dist.get_world_size(), shuffle=True
            )
            shuffle = False

        else:
            test_sampler = None
            shuffle = True

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_threads,
            pin_memory=False,
            drop_last=True,
            sampler=test_sampler,
        )

    return Data(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
    )


@ex.automain
def main(_run):
    from tqdm import tqdm
    from utils.tupperware import tupperware

    args = tupperware(_run.config)

    data = get_dataloaders(args)

    for _ in tqdm(data.train_loader.dataset):
        pass
