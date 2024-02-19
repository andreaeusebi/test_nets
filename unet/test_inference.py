# Allow Crtl-C to Works Despite Open Windows
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from euse_unet import EuseUnet
from model import UNET
import utils
import datasets.cityscapes_dataset as cityscapes_dataset

import logging

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Resize, InterpolationMode
from torchvision.transforms.functional import pil_to_tensor

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_IMG_PATH = ("/home/andrea/datasets/cityscapes/leftImg8bit/val/frankfurt/"
                  "frankfurt_000000_000294_leftImg8bit.png")

GT_IMG_PATH = ("/home/andrea/datasets/cityscapes/gtFine/val/frankfurt/"
               "frankfurt_000000_000294_gtFine_labelIds.png")

LOAD_MODEL = True
MODEL_PATH = "./model_params/UNET_params.pth"

def main():
    logging.basicConfig(format="[test_inference.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    logging.info("--- main() started. ---")

    # Open input image as PIL image and display it
    pil_img = Image.open(INPUT_IMG_PATH).convert("RGB")

    plt.figure()
    plt.imshow(pil_img)
    plt.axis(True)
    plt.title("Original unresized image.")

    # Convert to a torch tensor (it is important to use same
    # transform used in training, in this case ToTensor()).
    img = ToTensor()(pil_img)
    utils.logTensorInfo(img, "Original unresized image")

    # Resize
    img = Resize(size=(128, 256), antialias=False)(img)
    utils.logTensorInfo(img, "Resized Image")

    # Display resized image
    plt.figure()
    plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.axis(True)
    plt.title("Resized Image.")

    # Add a new dimension for batch size
    img = img.unsqueeze(dim=0)
    utils.logTensorInfo(img, "Image with batch dimension")

    # Send to GPU
    img = img.to(device)
    utils.logTensorInfo(img, "Image sent to GPU")

    ## Instantiate the model
    net = UNET(in_channels_=3,
               out_channels_=34)
    
    if LOAD_MODEL is True:
        # Load in the saved state_dict()
        net.load_state_dict(torch.load(f=MODEL_PATH))

    logging.info("### Model params correctly loaded! ###")

    # Send model to GPU
    net = net.to(device)
   
    net.eval()
    with torch.inference_mode():
        y_logits = net(img)

    utils.logTensorInfo(y_logits, "y_logits")
    
    ## Pass from logits to prediction labels
    y_pred_labels = torch.softmax(y_logits, dim=1).argmax(dim=1)
    utils.logTensorInfo(y_pred_labels, "y_pred_labels")

    ## Go back to CPU
    y_pred_labels = y_pred_labels.to("cpu")

    ## Plot with correct label colors
    y_pred_labels_cc = utils.labelToMask(y_pred_labels.squeeze(), cityscapes_dataset.PALETTE)
    utils.logTensorInfo(y_pred_labels_cc, "y_pred_labels_cc")

    plt.figure()
    plt.imshow(torch.permute(y_pred_labels_cc, (1, 2, 0)))
    plt.axis(True)
    plt.title("Predicted labels with correct label colors")

    ### Ground truth ###

    # Open gt image as PIL image and display it
    pil_gt_img = Image.open(GT_IMG_PATH)

    plt.figure()
    plt.imshow(pil_gt_img, cmap="gray", vmin=0, vmax=255)
    plt.axis(True)
    plt.title("Original Ground Truth image.")

    # Convert to a torch tensor (it is important to use same
    # transform used in training, in this case utils.PILToTensor()).
    gt_img = utils.PILToTensor()(pil_gt_img)
    utils.logTensorInfo(gt_img, "Original ground truth image")

    ## Resize (used nearest_exact to maintain correct labels)
    gt_img = Resize(size=(128, 256),
                    antialias=False,
                    interpolation=InterpolationMode.NEAREST_EXACT)(gt_img)
    utils.logTensorInfo(gt_img, "Ground truth image resized")

    # Display resized gt image
    plt.figure()
    plt.imshow(gt_img.squeeze(), cmap="gray", vmin=0, vmax=255)
    plt.axis(True)
    plt.title("Resized Ground Truth Image.")

    ## Convert GT labels to segmented color map
    gt_img_cc = utils.labelToMask(gt_img.squeeze(), cityscapes_dataset.PALETTE)

    plt.figure()
    plt.imshow(torch.permute(gt_img_cc, (1, 2, 0)))
    plt.axis(True)
    plt.title("Ground truth with coloured labels.")
    plt.show()

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()