## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from segments import SegmentsClient
from segments.huggingface import release2dataset
from segments.utils import get_semantic_bitmap
from huggingface_hub import login
from datasets import DatasetDict

import json

import config

def convert_segmentation_bitmap(example):
    return {
        "label.segmentation_bitmap":
            get_semantic_bitmap(
                example["label.segmentation_bitmap"],
                example["label.annotations"],
            )
    }

def convertSegBitmapsToUint8(example):
    example["label.segmentation_bitmap"] = example["label.segmentation_bitmap"].convert("L")

    return {
        "label.segmentation_bitmap":
            example["label.segmentation_bitmap"]
    }

def main():
    # Initialize a SegmentsDataset from the release file
    segments_client = SegmentsClient(config.SEGMENTS_API_KEY)
    segments_release = segments_client.get_release(
        config.SEGMENTS_DATASET_ID,
        config.SEGMENTS_DS_RELEASE
    )

    # Login to Huggingface
    login(config.HF_TOKEN)

    # Convert it to a Huggingface Dataset
    hf_dataset = release2dataset(segments_release, download_images=True)

    # The returned object is an Huggingface Dataset
    print(f"##### ----- hf_dataset:\n{hf_dataset}")

    # Keep only reviewed labeled images
    hf_dataset = hf_dataset.filter(lambda x: x["status"] == "REVIEWED")

    print(f"##### ----- hf_dataset filtered:\n{hf_dataset}")

    # Convert instances bitmap to segmentation bitmap
    semantic_dataset = hf_dataset.map(convert_segmentation_bitmap)
    
    #### THIS DOESN'T WORK ####
    # semantic_dataset = hf_dataset.map(convertSegBitmapsToUint8)

    # Divide the dataset into train and val splits
    splits_file = "tmhmi_ds_splits.json"
    ds_splits = json.load(open(splits_file))

    ds_splits = {k: v for k, v in ds_splits.items()}

    print(f"ds_splits: {ds_splits}")

    try:
        train_dataset = semantic_dataset.filter(lambda x: ds_splits[x["name"]] == "train")
        val_dataset = semantic_dataset.filter(lambda x: ds_splits[x["name"]] == "val")
    except KeyError as e:
        print(f"### ERROR ### Img '{e.args[0]}' isn't in the dict provided by '{splits_file}'!")
        exit(1)

    print(f"##### ----- Train split:\n{train_dataset}")
    print(f"##### ----- Val split:\n{val_dataset}")

    semantic_dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

    # Rearrange dataset columns
    semantic_dataset = semantic_dataset.rename_column('image', 'pixel_values')
    semantic_dataset = semantic_dataset.rename_column('label.segmentation_bitmap', 'label')
    semantic_dataset = semantic_dataset.remove_columns(
        ['name', 'uuid', 'status', 'label.annotations']
    )

    print(f"##### ----- Semantic dataset:\n{semantic_dataset}")

    print(f"##### ----- First row of train split:\n{semantic_dataset['train'][0]}")

    # Push to HF hub as private repo
    semantic_dataset.push_to_hub(config.HF_DATASET, private=True)
    
    print("Completed!")

if __name__ == "__main__":
    main()
