import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

""" 
U-Net implementation based on this tutorial:
    https://www.youtube.com/watch?v=IHq1t7NxS8k&t=1409s
"""

class DoubleConv(nn.Module):
    def __init__(self, in_channels_, out_channels_):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            ## 1st Convolution
            # Same Convolution (utput size is the same as the input size)
            nn.Conv2d(in_channels=in_channels_,
                      out_channels=out_channels_,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), # since we use batch norm
            nn.BatchNorm2d(out_channels_),
            nn.ReLU(inplace=True),

            ## 2nd Convolution
            nn.Conv2d(in_channels=out_channels_,
                      out_channels=out_channels_,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), # since we use batch norm
            nn.BatchNorm2d(out_channels_),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_):
        return self.conv(x_)

class UNET(nn.Module):
    def __init__(self,
                 in_channels_=3,
                 out_channels_=1,
                 features_=[64, 128, 256, 512]):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features_:
            self.downs.append(DoubleConv(in_channels_, feature))
            in_channels_ = feature

        # Up part of UNET
        for feature in reversed(features_):
            # Up convolution part of the up step
            self.ups.append(nn.ConvTranspose2d(
                in_channels=feature*2,
                out_channels=feature,
                kernel_size=2,
                stride=2)
            )

            # Double convolution part of the up step
            # (to be executed after padding of skip connection)
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottop part of UNET
        self.bottleneck = DoubleConv(features_[-1], features_[-1]*2)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(in_channels=features_[0],
                                    out_channels=out_channels_,
                                    kernel_size=1)
        
    def forward(self, x_ : torch.Tensor):
        """ 
        Image height and width must be divisible by two at every down step!
        To be precise, input height and width must be divisible by 16!
        If this is not verified, one or more feature maps may be resized
        during the up steps.
        """

        skip_connections = []

        for down in self.downs:
            x_ = down(x_)
            skip_connections.append(x_)

            x_ = self.pool(x_)

        x_ = self.bottleneck(x_)

        # Reverse the skip connections list
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # Up convolution
            x_ = self.ups[idx](x_)

            # Retrieve corresponding skip connection
            skip_connection = skip_connections[idx // 2]

            # If shapes are not equal it means that MaxPool has floored
            # since tensor was not divisible by 2.
            # Thus resize feature map to match skip connection h and w.
            if x_.shape != skip_connection.shape:
                x_ = TF.resize(x_, size=skip_connection.shape[2:])

            # Concat skip conn and current feature map along channel dim
            concat_skip = torch.cat((skip_connection, x_), dim=1)

            # Double convolution
            x_ = self.ups[idx+1](concat_skip)

        # Perfom final conv and then return
        return self.final_conv(x_)
    
def test():
    in_channels     = 1
    out_features    = 1

    x = torch.randn((3, in_channels, 128, 256))
    
    model = UNET(in_channels_=in_channels,
                 out_channels_=out_features)
    
    preds = model(x)

    ## Check wheter input and output shapes match
    print(x.shape)
    print(preds.shape)

    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
        