import os
from pathlib import Path
import logging
import torch

"""
Project folders structure:
project/
    models/         Where models state dictionaries and onnx files are saved.
    nets/           Contains networks implementation source code.
    config.py
    train.py
    demo.py
    demo.png
"""

##### ----- AUTOMATIC PARAMETERS ----- ######

DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR             = str(Path(os.path.dirname(__file__)).absolute()) +  "/"

################################################

##### ----- PATHS TO FILES & DIRECTORIES ----- ######

MODELS_DIR              = PROJECT_DIR + "models/"

################################################

##### ----- GENERIC PARAMETERS ----- ######

LOGGING_LEVEL           = logging.INFO

################################################

##### ----- INPUT PARAMETERS ----- ######

IN_MODEL_NAME           = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

H                       = 512
W                       = 1024
C                       = 3

NUM_CLASSES             = 19

################################################

##### ----- OUTPUT PARAMETERS ----- ######

OUT_MODEL_NAME          = "segformer-b0-finetuned-cityscapes-512-1024.pth"
OUT_ONNX_NAME           = "segformer-b0-finetuned-cityscapes-512-1024"

################################################

##### ----- TRAINING PARAMETERS ----- ######
################################################
