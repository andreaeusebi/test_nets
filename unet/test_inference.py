from euse_unet import EuseUnet
import utils

import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Resize

def main():
    logging.basicConfig(format="[test_inference.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.info("--- main() started. ---")

    ## Load training dataset
    # Use ToTensor() for 'transform' since we want tensor to be of float type
    train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="train",
                            mode="fine",
                            target_type="semantic",
                            transform=ToTensor(),
                            target_transform=utils.PILToTensor()) # returns uint8

    img_sample, smnt_sample = train_data[0]
    
    utils.logTensorInfo(img_sample, "img_sample")
    utils.logTensorInfo(smnt_sample, "smnt_sample")

    logging.info(f"img_sample[0, 5]: {img_sample[0, 5]}")
    logging.info(f"smnt_sample[0, 5]: {smnt_sample[0, 5]}")
    
    net = EuseUnet(input_channels_=3,
                   output_channels_=2)
    
    # Resize image to a [3x572x572] as expected by Unet
    resized_img = Resize(size=(572, 572), antialias=True)(img_sample)
    
    # Add a new dimension for batch size
    resized_img = resized_img.unsqueeze(dim=0)

    utils.logTensorInfo(resized_img, "resized_img")

    logit = net(resized_img)

    utils.logTensorInfo(logit, "logit")

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()