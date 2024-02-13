from utils import logTensorInfo

import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

def main():
    logging.basicConfig(format="[test_dataset_loading.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.debug("--- main() started. ---")

    ## Load training and validation dataset
    train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="train",
                            mode="fine",
                            target_type="semantic",
                            transform=ToTensor(),
                            target_transform=ToTensor())
    
    val_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                          split="val",
                          mode="fine",
                          target_type="semantic",
                          transform=ToTensor(),
                          target_transform=ToTensor())
    
    logging.debug(f"train data type: {type(train_data)}")
    logging.info(f"train images: {len(train_data.images)}")
    logging.info(f"train targets: {len(train_data.targets)}")
    logging.info(f"validation images: {len(val_data.images)}")
    logging.info(f"validation targets: {len(val_data.targets)}")
    logging.debug(f"Classes: {train_data.classes}")

    img_sample, smnt_sample = train_data[0]
    
    logTensorInfo(img_sample, "img_sample")
    
    logging.debug(f"smnt_sample type: {type(smnt_sample)}")
    logTensorInfo(smnt_sample, "smnt_sample")

    ## Plot first image with corresponding semantically segmented image
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(torch.permute(img_sample, (1, 2, 0)))
    plt.axis(False)
    fig.add_subplot(1, 2, 2)
    plt.imshow(smnt_sample.squeeze())
    plt.axis(False)
    plt.suptitle("First train image and segmentation (don't consider colormap)")
    plt.waitforbuttonpress()

    ## Plot more images
    torch.manual_seed(42)
    fig = plt.figure()
    rows, cols = 3, 3
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        img, smnt = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(torch.permute(img, (1, 2, 0)))
        plt.axis(False)

    plt.suptitle("Some random images from train set")
    plt.waitforbuttonpress()

    ## Prepare DataLoader
    # Setup the batch size hyperparameter
    BATCH_SIZE = 32
    
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data,               # dataset to turn into iterable
                                  batch_size=BATCH_SIZE,    # how many samples per batch? 
                                  shuffle=True)             # shuffle data every epoch?
    
    val_dataloader = DataLoader(val_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    # Let's check out what we've created
    logging.info(f"Dataloaders: {train_dataloader, val_dataloader}") 
    logging.info(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    logging.info(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")

    # Check out what's inside the training dataloader
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    logging.info(f"train_features_batch.shape: {train_features_batch.shape}")
    logging.info(f"train_labels_batch.shape: {train_labels_batch.shape}")

    logging.debug("--- main() completed. ---")

if __name__ == "__main__":
    main()