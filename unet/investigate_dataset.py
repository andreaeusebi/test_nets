import logging
from PIL import Image

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt

### Custom modules ###
import utils

## Path to labels image
GT_LABEL_IMG_PATH = ("/home/andrea/datasets/cityscapes/gtFine/train/aachen/"
                     "aachen_000046_000019_gtFine_labelIds.png")

## Path to directory where to save mask images
OTPUT_MASKS_PATH = ("/home/andrea/Pictures/image_segmentation/masks/")

## List of labels to be looked for
labels = list(range(-1, 34))

print(labels)

def main():
    """
    Create n mask images from a given image with labelled pixels.
    Each mask image isolate only the pixels belonging to its specific
    class.
    """
    
    logging.basicConfig(format="[investigate_dataset.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)
    
    logging.info("--- main() started. ---")

    logging.info(f"Input label img path: {GT_LABEL_IMG_PATH}")

    # Open input image as PIL image and display it
    pil_img = Image.open(GT_LABEL_IMG_PATH)
    pil_img.show()

    logging.info("Press a button to start processing...")
    input()

    # Convert to a torch tensor of the same type
    img = pil_to_tensor(pil_img)
    utils.logTensorInfo(img, "img")

    # Create a seprate image (mask) for each label class
    for label_idx in labels:
        mask = torch.ones(size=img.shape, dtype=img.dtype) * label_idx
        utils.logTensorInfo(mask, "mask")
        
        label_img = torch.eq(img, mask)
        print(label_img)

        label_img = label_img.type(torch.uint8) * 255
        print(label_img)

        pil_label_img = to_pil_image(label_img)

        pil_label_img = pil_label_img.save(OTPUT_MASKS_PATH + "mask_" +
                                           str(label_idx) + ".png")


    pil_img.close()

    logging.info("--- main() completed. ---")

if __name__ == "__main__":
    main()