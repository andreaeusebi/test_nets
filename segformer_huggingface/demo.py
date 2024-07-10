## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pylab as plt
import torch
import numpy as np
import argparse
import logging
from pathlib import Path

import config

def main():
    ## Collect global params to make visible which are expected by this script
    LOGGING_LEVEL = config.LOGGING_LEVEL
    IN_MODEL_NAME = config.IN_MODEL_NAME
    DEVICE        = config.DEVICE
    H             = config.H
    W             = config.W
    PALETTE       = config.PALETTE

    logging.basicConfig(format="[demo.py][%(levelname)s]: %(message)s",
					    level=LOGGING_LEVEL)
    
    ## Parse input arguments
    parser=argparse.ArgumentParser(description="Run a SegFormer model on a given image.")
    parser.add_argument("image_fpath", nargs='?', default="./demo.png")

    args=parser.parse_args()

    logging.debug(f"Input image: {args.image_fpath}")

    assert(Path(args.image_fpath).exists())

    ## Load the model
    model = SegformerForSemanticSegmentation.from_pretrained(IN_MODEL_NAME)
    model.to(DEVICE)

    ## Setup image processor (for both pre and post processing of images)
    processor = SegformerImageProcessor(size= {"height": H, "width": W})

    ## Load image
    image = Image.open(args.image_fpath).convert("RGB")
    
    ## Preprocess the image (resize + normalize)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    logging.info(f"pixel_values.shape: {pixel_values.shape}")

    ## Forward pass
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # The model outputs logits of shape (batch_size, num_labels, height/4, width/4)
    logging.info(f"logits.shape: {logits.shape}")

    ## Rescale logit to match image original and perform argmax
    predicted_segmentation_map = processor.post_process_semantic_segmentation(
        outputs, 
        target_sizes=[image.size[::-1]]
    )[0]

    logging.info(f"predicted_segmentation_map.shape: {predicted_segmentation_map.shape}")
    
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

    ## Get coloured mask from predicted labels mask
    color_seg = np.zeros((predicted_segmentation_map.shape[0],
                          predicted_segmentation_map.shape[1],
                          3),
                         dtype=np.uint8) # height, width, 3
   
    for label_id, color in PALETTE.items():
        color_seg[predicted_segmentation_map == label_id, :] = color
    
    ## Show image overlapped with colour mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    plt.figure(figsize=(15, 10))
    plt.imshow(color_seg)

    plt.show()

if __name__ == "__main__":
    main()
