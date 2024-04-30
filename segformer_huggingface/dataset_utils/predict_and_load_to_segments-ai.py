## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

## Add parent folder path to Python path
import sys 
sys.path.insert(0, "./../" )

import logging

from segments import SegmentsClient, SegmentsDataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from segments.utils import bitmap2file

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from segformer_huggingface import config

def visualize(*args):
    images = args
    for i, image in enumerate(images):
        plt.subplot(1,len(images),i+1)
        plt.imshow(np.array(image))
    plt.show()

def convertToSegmentsFormat(seg_bitmap: np.array):
    """
    Convert from segmentation bitmap to instance bitmap (NOT WORKING!),
    as well as create annotations list as expect by Segments.ai.
    """
    instance_bitmap = np.copy(seg_bitmap).astype(np.uint32)
    annotations = []
    
    instance_id = 1

    for label_id, color in config.PALETTE.items():
        category_id = label_id

        logging.info(f"Evaluating category id: {category_id+1}...")

        if (seg_bitmap == category_id).astype(np.int8).sum() == 0:
            logging.info("Skipping id")
            continue

        instance_bitmap[seg_bitmap == category_id] = instance_id

        logging.info(f"'id': {instance_id}, 'category_id': {category_id+1}")
        annotations.append({'id': instance_id, 'category_id': category_id+1})

        instance_id += 1

    return instance_bitmap, annotations


def main():
    logging.basicConfig(format="[predict_and_load_to_segments-ai.py][%(levelname)s]: %(message)s",
					    level=config.LOGGING_LEVEL)
    
    # Initialize a SegmentsDataset from the release file
    segments_client = SegmentsClient(config.SEGMENTS_API_KEY)
    segments_release = segments_client.get_release(
        config.SEGMENTS_DATASET_ID,
        config.SEGMENTS_DS_RELEASE
    )

    # Initialize a new dataset, this time containing only unlabeled images
    dataset = SegmentsDataset(segments_release, labelset='ground-truth', filter_by='unlabeled')

    ## Load the model
    model = SegformerForSemanticSegmentation.from_pretrained(config.IN_MODEL_NAME)
    model.to(config.DEVICE)

    ## Setup image processor (for both pre and post processing of images)
    processor = SegformerImageProcessor(size= {"height": config.H, "width": config.W})

    for sample in dataset:
        # Generate label predictions
        image = sample['image'].convert("RGB")

        logging.info(f"image type: {type(image)}")

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

        instance_bitmap, annotations = convertToSegmentsFormat(predicted_segmentation_map)

        ## Get coloured mask from predicted labels mask
        color_seg = np.zeros((predicted_segmentation_map.shape[0],
                              predicted_segmentation_map.shape[1],
                              3),
                             dtype=np.uint8) # height, width, 3

        for label_id, color in config.PALETTE.items():
            color_seg[predicted_segmentation_map == label_id, :] = color

        # Visualize the predictions
        visualize(image, color_seg, instance_bitmap)

        # Upload the predictions to Segments.ai
        file = bitmap2file(instance_bitmap)
        asset = segments_client.upload_asset(file, 'label.png')    
        attributes = {
            'format_version': '0.1',
            'annotations': annotations,
            'segmentation_bitmap': { 'url': asset.url },
        }
        segments_client.add_label(sample['uuid'], 'ground-truth', attributes, label_status='PRELABELED')

        logging.info(f"Loaded to Segments.ai!")

    exit()

    ## Load image
    image = Image.open(args.image_fpath).convert("RGB")
    
    ## Preprocess the image (resize + normalize)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(config.DEVICE)

    logging.info(f"pixel_values.shape: {pixel_values.shape}")
    
if __name__ == "__main__":
    main()