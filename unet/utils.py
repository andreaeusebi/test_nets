import logging

import torch
from torchvision.transforms import functional

def logTensorInfo(x_ : torch.Tensor, name_ : str):
    """
    Log tensor meta data to output.

    Args:
        x_  (torch.Tensor): Tensor whose metadata must be printed.

        name_ (str): Name of the tensor.
    """

    logging.info(f"--- {name_} Info:")
    logging.info(f"--- Shape: {x_.shape}")
    logging.info(f"--- Dtype: {x_.dtype}")
    logging.info(f"--- Device: {x_.device}")
    logging.info(f"------------------------")

class PILToTensor:
    """
    Class converting a PIL Image to a tensor of the same type. 
    """

    def __call__(self, image: any):
        """
        Call method. Convert a PIL Image to a tensor of the same type

        Args:
            image (PIL Image): Image to be converted to tensor.

        Returns:
            torch.Tensor: Converted image.
        """

        image = functional.pil_to_tensor(image)
        return image

def labelToMask(label: torch.Tensor, palette: list) -> torch.Tensor:
    """
    Transform a 2D label torch tensor into a 3D RGB tensor using colors
    encoded in the given palette. 

    Args:
        label (torch.Tensor): 2D tensor of integers representing the class to which
        each pixel belongs to (size is HxW).

        palette (list): List of 3D tuples (R,G,B) encoding the color for each
        specific label class.

    Returns:
        torch.Tensor: Tensor representing the coloured segmentation mask
        (size is 3xHxW, dtype=float32, each element is in the 0-1 range).
    """
    
    (h, w) = label.shape
    mask = torch.zeros((h, w, 3), dtype = torch.float32)
    for i, color in enumerate(palette):
        mask[label == i] = torch.tensor(color)/255.0
    return mask.permute(2, 0, 1)
