import logging
import random
from PIL import Image
import matplotlib.pyplot as plt

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

class PILToLongTensor:
    """
    Class converting a PIL Image to a Long tensor (torch.int64). 
    """

    def __call__(self, image: any):
        """
        Call method. Convert a PIL Image to a tensor of Long type

        Args:
            image (PIL Image): Image to be converted to tensor.

        Returns:
            torch.Tensor: Converted tensor image of type torch.int64.
        """

        image = functional.pil_to_tensor(image)
        image = image.type(torch.LongTensor)
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

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

    plt.show()
