## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from segments import SegmentsClient
from segments.huggingface import release2dataset
from segments.utils import get_semantic_bitmap
from huggingface_hub import login

import config

def convert_segmentation_bitmap(example):
    return {
        "label.segmentation_bitmap":
            get_semantic_bitmap(
                example["label.segmentation_bitmap"],
                example["label.annotations"],
            )
    }


def main():
    # Initialize a SegmentsDataset from the release file
    segments_client = SegmentsClient(config.SEGMENTS_API_KEY)
    segments_release = segments_client.get_release(
        f"{config.SEGMENTS_USERNAME}/{config.DATASET_ID}",
        config.DATASET_RELEASE
    )

    # Login to Huggingface
    login(config.HF_TOKEN)

    # Convert it to a Huggingface Dataset
    hf_dataset = release2dataset(segments_release, download_images=True)

    # The returned object is an Huggingface Dataset
    print(f"##### ----- hf_dataset:\n{hf_dataset}")

    # Filter out all examples which are not labeled!
    hf_dataset = hf_dataset.filter(lambda x: x["status"] != "UNLABELED" and x["status"] != "PRELABELED")

    print(f"##### ----- hf_dataset filtered:\n{hf_dataset}")

    # Convert instances bitmap to segmentation bitmap
    semantic_dataset = hf_dataset.map(convert_segmentation_bitmap)

    # Rearrange dataset columns
    semantic_dataset = semantic_dataset.rename_column('image', 'pixel_values')
    semantic_dataset = semantic_dataset.rename_column('label.segmentation_bitmap', 'label')
    semantic_dataset = semantic_dataset.remove_columns(
        ['name', 'uuid', 'status', 'label.annotations']
    )

    print(f"##### ----- Semantic dataset:\n{semantic_dataset}")

    # Push to HF hub as private repo
    semantic_dataset.push_to_hub(f"{config.HF_USERNAME}/{config.DATASET_ID}",
                                 private=True)
    
    print("Completed!")

if __name__ == "__main__":
    main()
