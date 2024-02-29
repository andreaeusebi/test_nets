import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from collections import namedtuple
from typing import Any, Optional, Callable, Tuple

from CustomCityscapesDataset import CustomCityscapesDataset
import cityscapes_dataset

import sys 
sys.path.insert(0, "./../" )
import utils

Label = namedtuple( "Label", [
    "name"          ,
    "id"            ,
    "color"         ,
    "originalIds"   ,
])

labels = [
    #       name               id       color               originalIds
    Label(  "background",       0,      (  0,  0,  0),      (  0,  1,  2,  3,  4,  5, 11, 12, 13, 14, 15, 16, 17, 18, 21, 23, -1) ),
    Label(  "flat",             1,      (128, 64,128),      (  6,  7,  8,  9, 10, 22)                                             ),
    Label(  "human",            2,      (220, 20, 60),      ( 24, 25)                                                             ),
    Label(  "vehicle",          3,      (  0,  0,142),      ( 26, 27, 28, 29, 30, 31, 32, 33)                                     ),
    Label(  "traffic_sign",     4,      (250,170, 30),      ( 19, 20)                                                             ),
]

PALETTE = [label.color for label in labels]

CLASS_WEIGHTS = [1.0 for label in labels]

print(CLASS_WEIGHTS)

class ReducedCityscapesDataset(CustomCityscapesDataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] =None) -> None:
        super().__init__(root, split, transform)

    ## Override the default __getitem__ method
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        # Call the superclass get item method to retrieve image and mask
        x, y = super().__getitem__(idx)

        # Remap the old labels to the new set of labels
        new_id = -1
        for original_label in cityscapes_dataset.labels:
            original_id = original_label.id
            for new_label in labels:
                if original_id in new_label.originalIds:
                    new_id = new_label.id
                    break
            if new_id == -1:
                new_id = 0
            
            y[y == original_id] = new_id

        return x, y
    
## Test the reduced dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    import torch

    ds_orig = CustomCityscapesDataset(root="/home/andrea/datasets/cityscapes",
                                      split="train")
    
    ds_reduced = ReducedCityscapesDataset(root="/home/andrea/datasets/cityscapes",
                                          split="train")
    
    ## Randomly get 3 pairs image and mask
    k = 3
    random.seed(42)
    img_idexes = random.sample(range(5000), k=k)

    for img_idx in img_idexes:
        img_orig, mask_orig = ds_orig[img_idx]
        img_red, mask_red = ds_reduced[img_idx]

        mask_orig_color = utils.labelToMask(mask_orig, cityscapes_dataset.PALETTE)
        mask_red_color = utils.labelToMask(mask_red, PALETTE)

        fig = plt.figure()
            
        fig.add_subplot(2, 2, 1)
        plt.imshow(img_orig)

        fig.add_subplot(2, 2, 2)
        plt.imshow(torch.permute(mask_orig_color, (1, 2, 0)))

        fig.add_subplot(2, 2, 3)
        plt.imshow(img_red)

        fig.add_subplot(2, 2, 4)
        plt.imshow(torch.permute(mask_red_color, (1, 2, 0)))

    plt.show()
