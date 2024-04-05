import torch

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_DIR         = "/home/andrea/personal_projects/test_nets/segformer_huggingface/"
WEIGTHS_FOLDER      = "weights/"

MODEL_WEIGHTS       = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

MODEL_NAME          = "segformer-b0-finetuned-cityscapes-1024-1024.pth"
ONNX_NAME           = "segformer-b0-finetuned-cityscapes-1024-1024"

H                   = 1080
W                   = 1920
C                   = 3

NUM_CLASSES         = 19

INPUT_IMG_PATH      = ("/home/andrea/phoenix_ws/src/phoenix_image_segmentation_pkg/"
                       "frame0000.jpg")
