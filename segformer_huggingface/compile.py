import torch
from transformers import SegformerForSemanticSegmentation

import config

def make_ONNX(model, file_out, example_input):
    torch.onnx.export(
        model,
        example_input,
        file_out + ".onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True
    )


EXAMPLE_INPUT = torch.randn(1, config.C, config.H, config.W, requires_grad=True, device=config.DEVICE)

# Build model

model = SegformerForSemanticSegmentation.from_pretrained(config.MODEL_WEIGHTS)

model.load_state_dict(torch.load(f=config.PROJECT_DIR + config.WEIGTHS_FOLDER + config.MODEL_NAME))

model.to(config.DEVICE)
model.eval()

# Compile and save model
make_ONNX(model, config.PROJECT_DIR + config.WEIGTHS_FOLDER + config.ONNX_NAME, EXAMPLE_INPUT)