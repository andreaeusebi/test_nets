import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

import utils
import datasets.cityscapes_dataset as cityscapes_dataset
from datasets.CustomCityscapesDataset import CustomCityscapesDataset

import logging
import random

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
import torchvision.transforms.v2 as TF
from torchvision.transforms.v2 import InterpolationMode

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

H               = 128
W               = 256
USE_CUSTOM      = True

def main():
    logging.basicConfig(format="[test_dataset_loading.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.info("--- main() started. ---")

    if USE_CUSTOM is True:
        train_transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomSizedCrop(min_max_height=(H*2, 1024/2),
                                          height=H,
                                          width=W,
                                          w2h_ratio=W/H,
                                          p=0.5,
                                          interpolation=3),
                        A.Resize(H, W, p=0.5, interpolation=3),
                    ],
                    p=1.0
                ),
                A.HorizontalFlip(p=0.25),
                # A.RGBShift(p=1.0),
                # A.Resize(H, W),
                # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.15),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.Resize(H, W, interpolation=3),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        train_data = CustomCityscapesDataset(
            root="/home/andrea/datasets/cityscapes",
            split="train",
            transform=train_transform)

        val_data = CustomCityscapesDataset(
            root="/home/andrea/datasets/cityscapes",
            split="val",
            transform=val_transform)

    else:
        # Write transform for image
        data_transform_augmentation = TF.Compose([
            # Resize the images
            TF.Resize(size=(H, W)),

            # Transforms using random transforms from a given list with random strength number.
            TF.TrivialAugmentWide(num_magnitude_bins=31), # how intense 

            # Turn the image into a torch.Tensor
            # this also converts all pixel values from 0 to 255 to 0.0 and 1.0
            TF.ToTensor() 
        ])

        # Create testing transform (no data augmentation)
        test_transform = TF.Compose([
            TF.Resize(size=(H, W)),
            TF.ToTensor()
        ])

        # Transform pipeline for target masks (both training and validation)
        target_transform = TF.Compose(
            [
                TF.Resize(size=(H, W),
                          interpolation=InterpolationMode.NEAREST_EXACT),
                utils.PILToLongTensor()
            ]
        )

        ## Load training and validation dataset
        train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                                split="train",
                                mode="fine",
                                target_type="semantic",
                                transform=data_transform_augmentation,
                                target_transform=target_transform)
        
        val_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="val",
                            mode="fine",
                            target_type="semantic",
                            transform=test_transform,
                            target_transform=target_transform)
    
    logging.info(f"Number of training data: {len(train_data)}")
    logging.info(f"train images: {len(train_data.images)}")
    logging.info(f"train targets: {len(train_data.targets)}")
    logging.info(f"validation images: {len(val_data.images)}")
    logging.info(f"validation targets: {len(val_data.targets)}")

    img_sample, smnt_sample = train_data[0]
    
    utils.logTensorInfo(img_sample, "img_sample")
    utils.logTensorInfo(smnt_sample, "smnt_sample")

    # Number of test images
    K = 5
    
    # Fix random seed so that we retrieve always the same images 
    random.seed(42)
    # for ii in random.sample(range(len(train_data)), k=5):
    for ii in range(K):
        img_sample, smnt_sample = train_data[ii]

        fig = plt.figure()
        
        fig.add_subplot(2, 1, 1)
        # Original image from dataset (after being subject to transform)
        plt.imshow(torch.permute(img_sample, (1, 2, 0)))
               
        # Corresponding coloured labels mask
        smnt_cc = utils.labelToMask(smnt_sample.squeeze(), cityscapes_dataset.PALETTE)
        fig.add_subplot(2, 1, 2)
        plt.imshow(torch.permute(smnt_cc, (1, 2, 0)))

    plt.show()
    exit()

    ## Plot first image with corresponding semantically segmented image
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(torch.permute(img_sample, (1, 2, 0)))
    fig.add_subplot(1, 2, 2)
    plt.imshow(smnt_sample.squeeze())
    plt.suptitle("First train image and segmentation (don't consider colormap)")

    ## Plot same semantic mask with original gray scale values
    plt.figure()
    plt.imshow(smnt_sample.squeeze(), cmap="gray", vmin=0, vmax=255)
    plt.title("Same semantic mask with original gray scale values.")

    ## Plot with correct label colors
    smnt_cc = utils.labelToMask(smnt_sample.squeeze(), cityscapes_dataset.PALETTE)

    utils.logTensorInfo(smnt_cc, "smnt_cc")

    plt.figure()
    plt.imshow(torch.permute(smnt_cc, (1, 2, 0)))
    plt.title("Plot with correct label colors")

    ## Plot more images
    torch.manual_seed(42)
    fig = plt.figure()
    rows, cols = 3, 3
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        img, smnt = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(torch.permute(img, (1, 2, 0)))

    plt.suptitle("Some random images from train set")

    ## Prepare DataLoader
    # Setup the batch size hyperparameter
    BATCH_SIZE = 32
    
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data,               # dataset to turn into iterable
                                  batch_size=BATCH_SIZE,    # how many samples per batch? 
                                  shuffle=True)             # shuffle data every epoch?
    
    val_dataloader = DataLoader(val_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    # Let's check out what we've created
    logging.info(f"Dataloaders: {train_dataloader, val_dataloader}") 
    logging.info(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    logging.info(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")

    # Check out what's inside the training dataloader
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    utils.logTensorInfo(train_features_batch, "train_features_batch")
    utils.logTensorInfo(train_labels_batch, "train_labels_batch")

    plt.show()

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()