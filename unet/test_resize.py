from utils import logTensorInfo

import logging

import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Resize

import matplotlib.pyplot as plt

def main():
    logging.basicConfig(format="[test_resize.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.info("--- main() started. ---")

    ## Load training dataset
    train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="train",
                            mode="fine",
                            target_type="semantic",
                            transform=ToTensor(),
                            target_transform=ToTensor())

    

    img_sample, smnt_sample = train_data[0]
    
    logTensorInfo(img_sample, "img_sample")

    ## Resize image to a [3x572x572] as expected by Unet
    resized_img = Resize(size=(572, 572), antialias=True)(img_sample)

    logTensorInfo(resized_img, "resized_img")

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(torch.permute(img_sample, (1, 2, 0)))
    plt.axis(False)
    fig.add_subplot(1, 2, 2)
    plt.imshow(torch.permute(resized_img, (1, 2, 0)))
    plt.axis(False)
    plt.suptitle("Original and resized")
    plt.waitforbuttonpress()

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()