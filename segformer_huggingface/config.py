import os
from pathlib import Path
import logging
import torch

from segformer_huggingface.dataset_utils.label import getPalette

"""
Project folders structure:
project/
    models/         Where models state dictionaries and onnx files are saved.
    nets/           Contains networks implementation source code.
    dataset_utils/  Contains util scripts and files for datasets handling.
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
SEGMENTS_DATASET_ID     = "andrea_eusebi/TMHMI_Semantic_Dataset"
SEGMENTS_DS_RELEASE     = "v0.7"
HF_USERNAME             = "eusandre95"
HF_TOKEN                = "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"
HF_DATASET              = "eusandre95/TMHMI_Semantic_Dataset"

################################################

##### ----- INPUT PARAMETERS ----- ######

# DATASETS = ["TMHMI", "Cityscapes"]. Is a list of datasets to use, first one defines labels set.
DATASETS                = ["TMHMI", "Cityscapes"]
PALETTE                 = getPalette(DATASETS[0])

IN_MODEL_NAME           = "nvidia/mit-b0"

H                       = 512
W                       = 512
C                       = 3

NUM_CLASSES             = 23

################################################

##### ----- OUTPUT PARAMETERS ----- ######

OUT_MODEL_NAME          = "240429-test-segformer-b0-finetuned-tmhmi-cs-512-1024-2ep-23labels"
OUT_ONNX_NAME           = "segformer-b0-finetuned-cityscapes-512-1024"

################################################

##### ----- TRAINING PARAMETERS ----- ######
################################################
