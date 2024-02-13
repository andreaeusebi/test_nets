from euse_unet import EuseUnet
from utils import logTensorInfo

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
    train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="train",
                            mode="fine",
                            target_type="semantic",
                            transform=ToTensor(),
                            target_transform=ToTensor())

    img_sample, smnt_sample = train_data[0]
    
    logTensorInfo(img_sample, "img_sample")
    logTensorInfo(smnt_sample, "smnt_sample")

    ## Prepare DataLoader
    # Setup the batch size hyperparameter
    BATCH_SIZE = 32
    
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data,               # dataset to turn into iterable
                                  batch_size=BATCH_SIZE,    # how many samples per batch? 
                                  shuffle=True)             # shuffle data every epoch?
    
    net = EuseUnet(input_channels_=3,
                   output_channels_=2)
    
    ## Resize image to a [3x572x572] as expected by Unet
    resized_img = Resize(size=(572, 572), antialias=True)(img_sample)

    logTensorInfo(resized_img.unsqueeze(dim=0), "resized_img")

    logit = net(resized_img.unsqueeze(dim=0))

    logTensorInfo(logit, "logit")

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()