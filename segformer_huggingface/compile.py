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

model = SegformerForSemanticSegmentation.from_pretrained(config.IN_MODEL_NAME)

model.to(config.DEVICE)
model.eval()

# Compile and save model
make_ONNX(model, config.MODELS_DIR + config.OUT_ONNX_NAME, EXAMPLE_INPUT)
