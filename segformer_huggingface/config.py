import os
from pathlib import Path
import logging
import torch

from dataset_utils.label import getPalette

"""
Project folders structure:
project/
    models/         Where models state dictionaries and onnx files are saved.
    nets/           Contains networks implementation source code.
    dataset_utils/  Contains util scripts and files for datasets handling.
    compile.py      Python script to compile a network and produce an .ONNX file.
    config.py       Python file containing common parameters.
    train.py        Python script for training a network.
    demo.py         Python script for quick inference testing.
    demo.png        Image used as default input for demo script.
"""

##### ----- AUTOMATIC PARAMETERS (DO NOT TOUCH) ----- ######

PROJECT_DIR             = str(Path(os.path.dirname(__file__)).absolute()) +  "/"
MODELS_DIR              = PROJECT_DIR + "models/"

################################################

##### ----- COMMON PARAMETERS ----- ######

LOGGING_LEVEL           = logging.INFO
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"

# Name of the pretrained network/backbone to use when loading the model from Huggingface Hub.
# You can find pretrained weights at the following link:
# https://huggingface.co/models?other=segformer
IN_MODEL_NAME           = "nvidia/mit-b0"
H                       = 512
W                       = 1024
C                       = 3

# DATASETS = ["TMHMI", "Cityscapes"]. Is a list of datasets to use, first one defines labels set and color palette.
DATASETS                = ["TMHMI", "Cityscapes"]
PALETTE                 = getPalette(DATASETS[0])

################################################

##### ----- DATASET PARAMETERS ----- ######

SEGMENTS_API_KEY        = "e6eb70a8f4cd51d900b5ca6a0fcbb504070b5307"
SEGMENTS_DATASET_ID     = "andrea_eusebi/TMHMI_Semantic_Dataset"
SEGMENTS_DS_RELEASE     = "v0.8"
HF_USERNAME             = "eusandre95"
HF_TOKEN                = "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"
HF_DATASET              = "eusandre95/TMHMI_Semantic_Dataset"

################################################

##### ----- TRAINING PARAMETERS ----- ######

EPOCHS                  = 50
LEARNING_RATE           = 0.00006
BATCH_SIZE              = 8

################################################

##### ----- OUTPUT PARAMETERS ----- ######

OUT_MODEL_NAME          = "240711-test"
OUT_ONNX_NAME           = "segformer-b0-finetuned-cityscapes-512-1024"

################################################
