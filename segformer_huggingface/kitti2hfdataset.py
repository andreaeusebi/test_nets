## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

import sys 
sys.path.insert(0, "./../" )

from huggingface_hub import login
from datasets import Dataset, DatasetDict
from PIL import Image

import os
import config

def main():

    # Login to Huggingface
    login(config.HF_TOKEN)

    # Prepare kitti dataset for pushing it into hf 
    # Initialize train and validation datasets
    train_dataset = {
                    'pixel_values' : [],
                    'label' : []
                    }   

    valid_dataset = {
                    'pixel_values' : [],
                    'label' : []
                    }
    
    # Open the file containing train split and get images paths
    split_files_dir = os.path.join(config.KITTI_DIR + "/data_2d_semantics/train")
    train_file = open(os.path.join(split_files_dir + "/2013_05_28_drive_train_frames.txt"))
    lines = train_file.readlines()
    train_file.close()

    for i, line in enumerate(lines):
        # Keep only one image every N images (Maybe in the future it will be better to consider the movement of the car)
        if i % config.N == 0:
            # Separate the paths of the raw data from the semantic ones and insert them in the dictionary
            line = line.split()
            train_dataset["pixel_values"].append(Image.open(os.path.join(config.KITTI_DIR, line[0])))
            train_dataset["label"].append(Image.open(os.path.join(config.KITTI_DIR, line[1])))

    # Repeat the process for the validation split
    valid_file = open(os.path.join(split_files_dir + "/2013_05_28_drive_val_frames.txt"))
    lines = valid_file.readlines()
    valid_file.close()

    for i, line in enumerate(lines):
        if i % config.N == 0:
            line = line.split()
            valid_dataset["pixel_values"].append(Image.open(os.path.join(config.KITTI_DIR, line[0])))
            valid_dataset["label"].append(Image.open(os.path.join(config.KITTI_DIR, line[1])))
    
    # Creation of Hugging face dataset
    hf_train_dataset = Dataset.from_dict(train_dataset)
    hf_valid_dataset = Dataset.from_dict(valid_dataset)
    kitti_hf_dataset = DatasetDict({"train": hf_train_dataset, "val": hf_valid_dataset})

    print(f"##### ----- Semantic dataset:\n{kitti_hf_dataset}")

    print(f"##### ----- First row of train split:\n{kitti_hf_dataset['train'][0]}")

    # Show the first image of the dataset
    kitti_hf_dataset['train'][0]["pixel_values"].show()
    kitti_hf_dataset['train'][0]["label"].show()

    # # Push to HF hub as private repo
    # kitti_hf_dataset.push_to_hub(config.HF_DATASET, private=True)
    
    print("Completed!")

if __name__ == "__main__":
    main()