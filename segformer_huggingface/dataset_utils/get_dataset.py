## Add parent folder path to Python path
import sys 
sys.path.insert(0, "./../" )

from typing import List, Dict
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets import Dataset, load_dataset, concatenate_datasets

from segformer_huggingface.dataset_utils.label import labelsIDsToTrainIdsOnArrays, LABELS_TMHMI, LABELS_CS
import segformer_huggingface.config as config

DEBUG_MODE                  = False
ACCEPTABLE_DATASETS_NAMES   = ["TMHMI", "Cityscapes"]
TMHMI_HF_DATASET_ID         = "eusandre95/TMHMI_Semantic_Dataset"
CITYSCAPES_HF_DATASET_ID    = "Antreas/Cityscapes"

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
            dataset = getHfTmhmiDataset(split_=split_)
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
        ds_concat = concatenate_datasets(datasets)

        logging.info(f"Concatenated dataset [split: '{split_}'] info:\n{ds_concat}")
        
        return ds_concat

def getHfTmhmiDataset(split_: str) -> Dataset:
    """
    Load TMHMI dataset from Huggingface and remap the labels to the trainIDs.
    """

    dataset = load_dataset(TMHMI_HF_DATASET_ID, split=split_)

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

    # Remap the labels to the train IDs
    dataset = dataset.map(remapTMHMILabels, writer_batch_size=100, num_proc=8)

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

def getHfCityscapesDataset(split_: str) -> Dataset:
    """
    Load Cityscapes dataset from Huggingface hub, perform the pre operations
    required to make it ready for Segformer and remap the labels to the trainIDs.
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

    # Remap the labels to the train IDs
    dataset = dataset.map(remapCityscapesLabels, writer_batch_size=100, num_proc=8)

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

def remapTMHMILabels(example):
    """
    Remap original TMHMI labels to the corresponding train IDs.
    Intended to be used as argument of a map() method.
    """

    orig_bitmap = np.array(example["label"])

    remapped_bitmap = labelsIDsToTrainIdsOnArrays(orig_bitmap, LABELS_TMHMI)

    return {
        "label": Image.fromarray(remapped_bitmap, mode="I")
    }

def remapCityscapesLabels(example):
    """
    Remap original Cityscapes labels to the corresponding train IDs.
    Intended to be used as argument of a map() method.
    """

    orig_bitmap = np.array(example["label"])

    remapped_bitmap = labelsIDsToTrainIdsOnArrays(orig_bitmap, LABELS_CS)

    return {
        "label": Image.fromarray(remapped_bitmap, mode="L")
    }

def test():
    logging.basicConfig(format="[dataset.py][%(levelname)s]: %(message)s",
					    level=logging.INFO)
    
    getHfDataset(dataset_names_=config.DATASETS, split_="train")

if __name__ == "__main__":
    test()
