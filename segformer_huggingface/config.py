import os
from pathlib import Path
import logging
import torch

from palettes import getPaletteFunction

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

##### ----- DATASET PARAMETERS ----- ######

SEGMENTS_API_KEY        = "e6eb70a8f4cd51d900b5ca6a0fcbb504070b5307"
SEGMENTS_USERNAME       = "andrea_eusebi"
DATASET_ID              = "TMHMI_Semantic_Dataset"
DATASET_RELEASE         = "v0.6"
HF_USERNAME             = "eusandre95"
HF_TOKEN                = "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"
HF_DATASET              = "Antreas/Cityscapes"

################################################

##### ----- INPUT PARAMETERS ----- ######

# DATASET = ["Cityscapes", "TMHMI"]
DATASET                 = "TMHMI"
PALETTE_FUNCTION        = getPaletteFunction(DATASET)

IN_MODEL_NAME           = "nvidia/mit-b0"

H                       = 512
W                       = 512
C                       = 3

NUM_CLASSES             = 19

################################################

##### ----- OUTPUT PARAMETERS ----- ######

OUT_MODEL_NAME          = "240423-segformer-b0-finetuned-cs-512-512-2ep-19labels"
OUT_ONNX_NAME           = "segformer-b0-finetuned-cityscapes-512-1024"

################################################

##### ----- TRAINING PARAMETERS ----- ######
################################################
