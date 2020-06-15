import torch
from torch import nn

from segmentation_config import Config


class SegmentationNet(nn.Module):
    """Segmentation network."""

    def __init__(self, config: Config):
        super().__init__()
        H, W = config.image_size
        if H % 8 or W % 8:
            raise IOError("Wrong input size")

        self.last_conv = nn.Conv2d(in_channels=1024, out_channels=1,
                                   kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.last_conv.weight)
        self.last_layernorm = nn.LayerNorm(
            (H // 2**3, W // 2**3))  # default eps=1e-05
        nn.init.ones_(self.last_layernorm.weight)
        nn.init.zeros_(self.last_layernorm.bias)
        self.relu = nn.ReLU()

        self.segmentation = nn.Sequential(
            ConvLayerNormReLU((1, H, W), 32, 5, 2),
            nn.MaxPool2d(2),
            ConvLayerNormReLU((32, H // 2, W // 2), 64, 5, 2),
            ConvLayerNormReLU((64, H // 2, W // 2), 64, 5, 2),
            ConvLayerNormReLU((64, H // 2, W // 2), 64, 5, 2),
            nn.MaxPool2d(2),
            ConvLayerNormReLU((64, H // 2**2, W // 2**2), 64, 5, 2),
            ConvLayerNormReLU((64, H // 2**2, W // 2**2), 64, 5, 2),
            ConvLayerNormReLU((64, H // 2**2, W // 2**2), 64, 5, 2),
            ConvLayerNormReLU((64, H // 2**2, W // 2**2), 64, 5, 2),
            nn.MaxPool2d(2),
            ConvLayerNormReLU((64, H // 2**3, W // 2**3), 1024, 15, 7),
            self.last_conv,
            self.last_layernorm
        )

        self.reduce_mask = ReduceMask()

    def forward(self, image, mask):
        segmentation = self.segmentation(image)
        reduced_mask = self.reduce_mask(mask)
        return segmentation, reduced_mask


class ConvLayerNormReLU(nn.Module):

    def __init__(self, x_size, out_channels, kernel_size, padding):
        super().__init__()
        # x_size: channels x height x width
        self.conv = nn.Conv2d(in_channels=x_size[0], out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        # normalize each channel separately
        self.layernorm = nn.LayerNorm(x_size[1:])  # default eps=1e-05
        nn.init.ones_(self.layernorm.weight)
        nn.init.zeros_(self.layernorm.bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        after_conv = self.conv(x)
        after_layernorm = self.layernorm(after_conv)
        after_relu = self.relu(after_layernorm)
        return after_relu


class ReduceMask(nn.Module):
    """This serves to adjust the size of the segmentation mask
    to the size of the SegmentationNet output.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)

    def forward(self, x):
        # this further increases the number of pixels with values
        # 0 < pixel value < 1
        return self.pool(x)
