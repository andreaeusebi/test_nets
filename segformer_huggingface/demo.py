from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pylab as plt
import torch
import numpy as np
import argparse
import logging
from pathlib import Path

import config

def cs_palette():
    """Cityscapes palette that maps each class to RGB values."""
    return [[128, 64,128], #  0 
            [244, 35,232], #  1
            [ 70, 70, 70], #  2
            [102,102,156], #  3
            [190,153,153], #  4
            [153,153,153], #  5
            [250,170, 30], #  6
            [220,220,  0], #  7
            [107,142, 35], #  8
            [152,251,152], #  9
            [ 70,130,180], # 10
            [220, 20, 60], # 11
            [255,  0,  0], # 12
            [  0,  0,142], # 13
            [  0,  0, 70], # 14
            [  0, 60,100], # 15
            [  0, 80,100], # 16
            [  0,  0,230], # 17
            [119, 11, 32]  # 18
           ]

def main():
    logging.basicConfig(format="[demo.py][%(levelname)s]: %(message)s",
					    level=config.LOGGING_LEVEL)
    
    ## Parse input arguments
    parser=argparse.ArgumentParser(description="Run a SegFormer model on a given image.")
    parser.add_argument("image_fpath", nargs='?', default="./demo.png")

    args=parser.parse_args()

    logging.debug(f"Input image: {args.image_fpath}")

    assert(Path(args.image_fpath).exists())

    ## Load the model
    model = SegformerForSemanticSegmentation.from_pretrained(config.IN_MODEL_NAME)
    model.to(config.DEVICE)

    ## Setup image processor (for both pre and post processing of images)
    processor = SegformerImageProcessor(size= {"height": config.H, "width": config.W})

    ## Load image
    image = Image.open(args.image_fpath).convert("RGB")
    
    ## Preprocess the image (resize + normalize)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(config.DEVICE)

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

    palette = np.array(cs_palette())
    
    for label, color in enumerate(palette):
        color_seg[predicted_segmentation_map == label, :] = color
    
    ## Show image overlapped with colour mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    plt.show()

if __name__ == "__main__":
    main()
