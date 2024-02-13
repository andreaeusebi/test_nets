import logging
import torch

def logTensorInfo(x_ : torch.Tensor, name_ : str):
    logging.info(f"--- {name_} Info:")
    logging.info(f"--- Shape: {x_.shape}")
    logging.info(f"--- Dtype: {x_.dtype}")
    logging.info(f"--- Device: {x_.device}")
    logging.info(f"------------------------")
