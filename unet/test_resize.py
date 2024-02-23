import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

import logging
import sys

import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Resize, InterpolationMode

import matplotlib.pyplot as plt

import utils

H_DEFAULT = 128
W_DEFAULT = 256

def main(argv):
    logging.basicConfig(format="[test_resize.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.info("--- main() started. ---")

    new_h = 0
    new_w = 0

    if len(argv) == 2:
        logging.info(f"Received args: {argv[0]}, {argv[1]}")
        new_h = int(argv[0])
        new_w = int(argv[1])
    elif len(argv) == 0:
        logging.info(f"Using default height and widht: {H_DEFAULT}, {W_DEFAULT}")
        new_h = H_DEFAULT
        new_w = W_DEFAULT
    else:
        logging.critical(f"2 expected args. Received {len(argv)}.")
        exit()

    logging.debug(f"new_h: {new_h}")
    logging.debug(f"new_w: {new_w}")

    ## Load training dataset
    train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="train",
                            mode="fine",
                            target_type="semantic",
                            transform=ToTensor(),
                            target_transform=utils.PILToLongTensor())

    img_sample, smnt_sample = train_data[0]
    
    utils.logTensorInfo(img_sample, "img_sample")
    utils.logTensorInfo(smnt_sample, "smnt_sample")

    ## Resize image to to the requested shape by Unet
    ## Disable antialiasing since it smooths edges (it adds ficticious classes
    ## at the borders of different labels)
    resized_img = Resize(size=(new_h, new_w), antialias=False)(img_sample)
    
    # For labels use nearest_exact to avoid edge smoothing
    resized_smnt = Resize(size=(new_h, new_w),
                          antialias=False,
                          interpolation=InterpolationMode.NEAREST_EXACT)(smnt_sample)

    utils.logTensorInfo(resized_img, "resized_img")
    utils.logTensorInfo(resized_smnt, "resized_smnt")

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(torch.permute(img_sample, (1, 2, 0)))
    plt.axis(False)
    fig.add_subplot(1, 2, 2)
    plt.imshow(torch.permute(resized_img, (1, 2, 0)))
    plt.axis(False)
    plt.suptitle("Original and resized")

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(torch.permute(smnt_sample, (1, 2, 0)).squeeze())
    plt.axis(False)
    fig.add_subplot(1, 2, 2)
    plt.imshow(torch.permute(resized_smnt, (1, 2, 0)).squeeze())
    plt.axis(False)
    plt.suptitle("Original and resized Segmenteds")

    logging.info("--- main() completed. ---")

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])