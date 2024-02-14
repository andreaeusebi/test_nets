import logging

import torch
from torchvision.transforms import functional

def logTensorInfo(x_ : torch.Tensor, name_ : str):
    logging.info(f"--- {name_} Info:")
    logging.info(f"--- Shape: {x_.shape}")
    logging.info(f"--- Dtype: {x_.dtype}")
    logging.info(f"--- Device: {x_.device}")
    logging.info(f"------------------------")

class PILToTensor:
    def __call__(self, image):
        image = functional.pil_to_tensor(image)
        return image
