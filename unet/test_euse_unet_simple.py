from euse_unet import EuseUnet

import logging

import torch

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    logging.basicConfig(format="[test_euse_unet_simple.py][%(levelname)s]: %(message)s",
                        level=logging.DEBUG)

    logging.debug("--- main() started. ---")

    logging.debug(f"Torch version: {torch.__version__}")
    logging.debug(f"Cuda available: {torch.cuda.is_available()}")

    ## Params
    input_channels = 1

    ## Instantiate an EuseUnet
    euse_unet = EuseUnet(input_channels_=input_channels).to(device)

    logging.debug(f"Model is on device: {next(euse_unet.parameters()).device}")

    ## Create a sample tensor with proper size and shape
    x_sample = torch.rand(size=(1, input_channels, 572, 572))
    x_sample = x_sample.to(device)

    logging.debug(f"x_sample is on device: {x_sample.device}")

    logging.debug(f"x_sample shape: {x_sample.shape}")
    logging.debug(f"First 5 elements: {x_sample[0, 0, 0, :5]}")

    ## Perform forward pass to test shapes
    y = euse_unet(x_sample)

    logging.debug("Forward pass completed!")

    logging.debug("--- main() completed. ---")

if __name__ == "__main__":
    main()