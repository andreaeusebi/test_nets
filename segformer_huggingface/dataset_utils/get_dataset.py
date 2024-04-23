from typing import List, Dict
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets import Dataset, load_dataset

DEBUG_MODE                  = False
ACCEPTABLE_DATASETS_NAMES   = ["TMHMI", "Cityscapes"]
CITYSCAPES_HF_DATASET_ID    = "Antreas/Cityscapes"

CS_TO_TMHMI_MAPPING = {
# CS_ID   TMHMI_ID   # TMHMI_ID
    0:      255,     #    0,   
    1:      255,     #    0,   
    2:      255,     #    0,   
    3:      255,     #    0,   
    4:      255,     #    0,   
    5:      255,     #   23,   
    6:      255,     #    0,   
    7:        0,     #    1,   
    8:        1,     #    2,   
    9:      255,     #    0,   
   10:      255,     #    0,   
   11:        2,     #    3,   
   12:        3,     #    4,   
   13:        4,     #    5,   
   14:      255,     #    0,   
   15:      255,     #    0,   
   16:      255,     #    0,   
   17:        5,     #    6,   
   18:      255,     #    0,   
   19:        6,     #    7,   
   20:        7,     #    8,   
   21:        8,     #    9,   
   22:        9,     #   10,   
   23:       10,     #   11,   
   24:       11,     #   12,   
   25:       12,     #   13,   
   26:       13,     #   14,   
   27:       14,     #   15,   
   28:       15,     #   16,   
   29:      255,     #   15,   
   30:      255,     #   15,   
   31:       16,     #    0,   
   32:       17,     #   18,   
   33:       18      #    0   
}

def getHfDataset(
        dataset_names_: List[str],
        split_: str
    ) -> Dataset:
    """
    Load and return an Huggingface dataset, possibly made of multiple datasets.
    
    Args:
        dataset_names_ List[str]: Names of the datasets to load.
        split_ [str]: The split to load ["train" or "val"].

    Return:
        The resulting Huggingface dataset.
    """

    # Check that the given names are valid
    assert(dataset_name in ACCEPTABLE_DATASETS_NAMES for dataset_name in dataset_names_)

    # List of Huggingface datasets
    datasets = []

    for dataset_name in dataset_names_:
        if dataset_name == "TMHMI":
            pass
        elif dataset_name == "Cityscapes":
            dataset = getHfCityscapesDataset(split_=split_)

        logging.info(f"Dataset '{dataset_name}' [split: '{split_}'] info:\n{dataset}")

        for ft_name, ft_value in dataset[0].items():
            logging.info(f"Feature '{ft_name}' is of type: {ft_value}")

        datasets.append(dataset)

    if len(datasets) == 1: # Single dataset
        return datasets[0]
    else:
        # Concatenate all the datasets
        pass

def getHfTmhmiDataset(split_: str) -> Dataset:
    """
    Load TMHMI dataset from Huggingface.
    """

    ##### ----- TODO ----- #####

    pass

def getHfCityscapesDataset(split_: str) -> Dataset:
    """
    Load Cityscapes dataset from Huggingface hub, perform the pre operations
    required to make it ready for Segformer and remap the labels to match
    the TMHMI labels IDs.
    """

    dataset = load_dataset(CITYSCAPES_HF_DATASET_ID, split=split_)

    # Rename the columns as expected by SegFormer
    dataset = dataset.rename_column("image", "pixel_values")
    dataset = dataset.rename_column("semantic_segmentation", "label")

    ### ----- DEBUG ------ ###
    if DEBUG_MODE:
        image = np.array(dataset[0]["pixel_values"])
        sem_bitmap = np.array(dataset[0]["label"])

        logging.info(f"##### ----- MIN: {np.min(sem_bitmap)}")
        logging.info(f"##### ----- MAX: {np.max(sem_bitmap)}")

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(image)
        fig.add_subplot(2, 1, 2)
        plt.imshow(sem_bitmap)
    ### ----- DEBUG ------ ###

    # Remap the label IDs from Cityscapes to TMHMI ones
    dataset = dataset.map(remapCStoTMHMILabels, writer_batch_size=100, num_proc=8)

    ### ----- DEBUG ------ ###
    if DEBUG_MODE:
        image = np.array(dataset[0]["pixel_values"])
        sem_bitmap = np.array(dataset[0]["label"])

        logging.info(f"##### ----- MIN: {np.min(sem_bitmap)}")
        logging.info(f"##### ----- MAX: {np.max(sem_bitmap)}")

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(image)
        fig.add_subplot(2, 1, 2)
        plt.imshow(sem_bitmap)
    
        plt.show()
    ### ----- DEBUG ------ ###

    return dataset

def remapCStoTMHMILabels(example):
    """
    Mapping fuction to map from Cityscapes labels to TMHMI labels.
    Intended to be used as argument of a map() method.
    """

    orig_bitmap = np.array(example["label"])

    remapped_bitmap = remapLabels(orig_bitmap, CS_TO_TMHMI_MAPPING)

    return {
        "label": Image.fromarray(remapped_bitmap, mode="L")
    }

def remapLabels(semantic_bitmap_: np.array,
                label_ids_mapping: Dict[int, int]):
    """ 
    Remap the labels IDs of the given segmentation bitmap using
    the given mapping dictionary.
    """
    
    # Initialize new semantic bitmap as a copy of the original one
    mapped_semantic_bitmap = np.copy(semantic_bitmap_)

    ## Remap the original labels to the new set of labels
    for orig_id, mapped_id in label_ids_mapping.items():            
        mapped_semantic_bitmap[semantic_bitmap_ == orig_id] = mapped_id

    return mapped_semantic_bitmap

def test():
    logging.basicConfig(format="[dataset.py][%(levelname)s]: %(message)s",
					    level=logging.INFO)
    
    getHfDataset(dataset_names_=["Cityscapes"], split_="train")

if __name__ == "__main__":
    test()
