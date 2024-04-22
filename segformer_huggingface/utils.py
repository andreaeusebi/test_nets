from typing import Dict
import numpy as np

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
