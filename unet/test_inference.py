## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

## Standard modules
import logging
import random

## Torch imports
import torch

## Other modules
import matplotlib.pyplot as plt

## Custom (Project) imports
from model import UNET
import utils
# from datasets.cityscapes_dataset import PALETTE
from datasets.ReducedCityscapesDataset import PALETTE
from datasets.ReducedCityscapesDataset import ReducedCityscapesDataset as Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

##### --- Params --- #####

H = 128
W = 256
K = 5       ## number of images on which testing inference
MODEL_PATH = "./model_params/UNET_params_240403_256w_34classes_to5_cs_20ep.pth"

##########################

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

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

    ## Define transformation pipeline (should be moved to a separate file)
    ## Remark: this should be the same applied during training!!!
    val_transform = A.Compose(
        [
            A.Resize(H, W, interpolation=3),
            ToTensorV2(),
        ]
    )

    # Retrieve the validation dataset
    val_data = Dataset(root="/home/andrea/datasets/cityscapes",
                       split="val",
                       transform=val_transform)
    
    # Fix random seed so that we retrieve always the same images 
    random.seed(42)
    # for ii in random.sample(range(len(train_data)), k=5):
    for ii in range(K):
        img, mask = val_data[ii]

        img = img.to(torch.float32)
        mask = mask.to(torch.int64)

        # Add a new dimension for batch size
        img = img.unsqueeze(dim=0)

        # Send to GPU
        img = img.to(device)

        net.eval()
        with torch.inference_mode():
            y_logits = net(img)

        # Pass from logits to prediction labels
        y_pred_labels = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # Go back to CPU
        y_pred_labels = y_pred_labels.to("cpu")

        # Convert to color mask
        y_pred_labels_cc = utils.labelToMask(y_pred_labels.squeeze(), PALETTE)

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(torch.permute(y_pred_labels_cc, (1, 2, 0)))
        plt.title("Predicted mask image")

        ## Convert GT to color mask
        gt_img_cc = utils.labelToMask(mask.squeeze(), PALETTE)

        fig.add_subplot(2, 1, 2)
        plt.imshow(torch.permute(gt_img_cc, (1, 2, 0)))
        plt.title("Ground truth mask image")

        fig.suptitle("Image Segmentation Results", fontsize=16)

    plt.show()

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()
