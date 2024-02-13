import logging

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop

class UpStep(nn.Module):
    def __init__(self, in_channels_, out_channels_):
        super().__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels_,
                                             in_channels_ // 2,
                                             kernel_size=2,
                                             stride=2)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_,
                      out_channels=out_channels_,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels_,
                      out_channels=out_channels_,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

    def forward(self, x1_, x2_):
        ## Upsampling and channels halving
        x1_ = self.upsampling(x1_)

        logging.debug(f"Shape after upsampling: {x1_.shape}")

        # ## Pad the smaller image to fit larger image shape
        # # input is CHW
        # diffY = x2_.size()[2] - x1_.size()[2]
        # diffX = x2_.size()[3] - x1_.size()[3]

        # x1_ = F.pad(x1_,
        #             [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        #             value=0.0)

        logging.debug(f"x2 shape: {x2_.shape}")

        ## Crop larger image to fit smaller image (as in the paper)
        x2_cropped = CenterCrop(size=(x1_.shape[2], x1_.shape[3]))(x2_)

        logging.debug(f"x2_cropped shape: {x2_cropped.shape}")
        
        ## Concat the two tensors
        x_cat = torch.cat([x2_cropped, x1_], dim=1)

        logging.debug(f"x_cat shape: {x_cat.shape}")

        ## Double conv
        return self.double_conv(x_cat)



class EuseUnet(nn.Module):
    """
    My implementation of the U-net CNN from paper
    "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    """

    def __init__(self, input_channels_ : int):
        """
        Init method.

        Args:
            input_channels_ (int): Number of channels of input images.
        """

        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels_,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

        self.block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

        self.block_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU()
        )

        self.block_6 = UpStep(in_channels_=1024,
                              out_channels_=512)
        
        self.block_7 = UpStep(in_channels_=512,
                              out_channels_=256)
        
        self.block_8 = UpStep(in_channels_=256,
                              out_channels_=128)
        
        self.block_9 = UpStep(in_channels_=128,
                              out_channels_=64)

        logging.debug(f"--- EusenUnet:__init__() completed. ---")

    def forward(self, x_ : torch.Tensor):
        """
        Overwritten forward method to perform inference.

        Args:
            x_ (torch.Tensor): Input image on which performing semantic segmentation.
            Size must be [N x 1 x 572 x 572] ([NCHW]).

        Returns:
            torch.Tensor: Output segmentation map. Size is [N x Z x 388 x 388]
        """

        x1 = self.block_1(x_)

        logging.debug(f"Shape after block 1: {x1.shape}")

        x2 = self.block_2(x1)

        logging.debug(f"Shape after block 2: {x2.shape}")

        x3 = self.block_3(x2)

        logging.debug(f"Shape after block 3: {x3.shape}")

        x4 = self.block_4(x3)

        logging.debug(f"Shape after block 4: {x4.shape}")

        x5 = self.block_5(x4)

        logging.debug(f"Shape after block 5: {x5.shape}")

        x = self.block_6(x5, x4)

        logging.debug(f"Shape after block 6: {x.shape}")

        x = self.block_7(x, x3)

        logging.debug(f"Shape after block 7: {x.shape}")

        x = self.block_8(x, x2)

        logging.debug(f"Shape after block 8: {x.shape}")

        x = self.block_9(x, x1)

        logging.debug(f"Shape after block 9: {x.shape}")

        return x

def main():
    logging.basicConfig(format="[euse_unet.py][%(levelname)s]: %(message)s",
                        level=logging.DEBUG)

    logging.debug("--- main() started. ---")

    logging.debug(f"Torch version: {torch.__version__}")
    logging.debug(f"Cuda available: {torch.cuda.is_available()}")

    euse_unet = EuseUnet(input_channels_=1)

    logging.debug("--- main() completed. ---")

if __name__ == "__main__":
    main()