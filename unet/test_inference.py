import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from euse_unet import EuseUnet
from model import UNET
import utils
import datasets.cityscapes_dataset as cityscapes_dataset
import datasets.cityscapes_utils as cs_utils

import logging
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Resize, InterpolationMode
from torchvision.transforms.functional import pil_to_tensor

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

H = 256
W = 512
K = 5       ## number of images on which testing inference

INPUT_IMG_PATH = "/home/andrea/datasets/cityscapes/leftImg8bit/"
MASK_IMG_PATH = "/home/andrea/datasets/cityscapes/gtFine/"
MODEL_PATH = "./model_params/UNET_params_24027_512w_50ep_album.pth"

def main():
    logging.basicConfig(format="[test_inference.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.info("--- main() started. ---")

    ## Instantiate the model
    net = UNET(in_channels_=3,
               out_channels_=34)
    
    # Load in the saved state_dict()
    net.load_state_dict(torch.load(f=MODEL_PATH))

    logging.info("### Model params correctly loaded! ###")

    # Send model to GPU
    net = net.to(device)

    paths = cs_utils.findRandomImagesAndMasksPaths(images_path_=INPUT_IMG_PATH,
                                                   masks_path_=MASK_IMG_PATH,
                                                   n_=K,
                                                   seed_=41)
    
    print(paths)

    ## Define transformation pipeline (should be moved to a separate file)
    ## Remark: this should be the same applied during training!!!
    val_transform = A.Compose(
        [
            A.Resize(H, W, interpolation=3),
            ToTensorV2(),
        ]
    )

    for img_path, mask_path in paths:
        
        # Open input image and mask as PIL images
        pil_img = Image.open(img_path).convert("RGB")
        pil_mask = Image.open(mask_path)

        ## Convert to numpy array
        np_img = np.array(pil_img)
        np_mask = np.array(pil_mask)

        ## Apply transformation pipeline
        transformed = val_transform(image=np_img, mask=np_mask)

        img = transformed["image"]
        mask = transformed["mask"]

        fig = plt.figure()
        
        fig.add_subplot(2, 2, 1)
        plt.imshow(pil_img)
        plt.title("Original RGB image")

        fig.add_subplot(2, 2, 2)
        plt.imshow(torch.permute(img, (1, 2, 0)))
        plt.title("Transformed (and reside) RGB image")

        img = img.to(torch.float32)
        mask = mask.to(torch.int64)

        # Add a new dimension for batch size
        img = img.unsqueeze(dim=0)

        # Send to GPU
        img = img.to(device)
   
        net.eval()
        with torch.inference_mode():
            y_logits = net(img)

        ## Pass from logits to prediction labels
        y_pred_labels = torch.softmax(y_logits, dim=1).argmax(dim=1)

        ## Go back to CPU
        y_pred_labels = y_pred_labels.to("cpu")

        ## Plot with correct label colors
        y_pred_labels_cc = utils.labelToMask(y_pred_labels.squeeze(), cityscapes_dataset.PALETTE)

        fig.add_subplot(2, 2, 3)
        plt.imshow(torch.permute(y_pred_labels_cc, (1, 2, 0)))
        plt.title("Predicted mask image")

        ## Convert GT labels to segmented color map
        gt_img_cc = utils.labelToMask(mask.squeeze(), cityscapes_dataset.PALETTE)

        fig.add_subplot(2, 2, 4)
        plt.imshow(torch.permute(gt_img_cc, (1, 2, 0)))
        plt.title("Ground truth mask image")

        fig.suptitle("Image Segmentation Results", fontsize=16)

    plt.show()

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()