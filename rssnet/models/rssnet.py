"""Script defining the RSS-Net architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => ReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => ReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ASPPBlock(nn.Module):
    """ Parallel conv blocks with different dilation rate"""
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class RSSNet(nn.Module):
    """RSSNet architecture

    PARAMETERS
    ----------
    nb_classes: int
    n_channels: int
        Number of input channels
    """

    def __init__(self, nb_classes, n_channels):
        super().__init__()
        self.nb_classes = nb_classes
        self.n_channels = n_channels
        self.double_conv_block1 = DoubleConvBlock(in_ch=self.n_channels, out_ch=48, k_size=3,
                                                  pad=2, dil=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=48, out_ch=128, k_size=3, pad=2, dil=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=48, k_size=1, pad=0, dil=1)
        self.double_conv_block3 = DoubleConvBlock(in_ch=128, out_ch=256, k_size=3, pad=2, dil=2)
        self.double_conv_block4 = DoubleConvBlock(in_ch=256, out_ch=512, k_size=3, pad=2, dil=2)
        self.aspp_block = ASPPBlock(in_ch=512, out_ch=256)
        self.single_conv_block2_1x1 = ConvBlock(in_ch=1280, out_ch=256, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch=304, out_ch=256, k_size=3, pad=1, dil=1)
        self.single_conv_block2_3x3 = ConvBlock(in_ch=256, out_ch=256, k_size=3, pad=1, dil=1)
        self.final_conv = nn.Conv2d(256, nb_classes, kernel_size=3, padding=0, dilation=1)

    def forward(self, x):
        x_width, x_height = x.shape[2:]
        # Backbone
        x1 = self.double_conv_block1(x)
        x1 = self.max_pool(x1)
        x2 = self.double_conv_block2(x1)
        x2 = self.max_pool(x2)

        # Use for ch-wise concat
        x3 = self.single_conv_block1_1x1(x2)

        # Before ASPP block
        x4 = self.double_conv_block3(x2)
        # x4 = self.max_pool(x4)
        x5 = self.double_conv_block4(x4)
        # x5 = self.max_pool(x5)

        # ASPP block
        x6 = self.aspp_block(x5)

        # Residual connexion
        # x7 = F.interpolate(self.single_conv_block2_1x1(x6), size=(100, 250),
                           # align_corners=False, mode='bilinear')
        # x8 = torch.cat((x3, x7), 1)
        x7 = self.single_conv_block2_1x1(x6)
        x8 = torch.cat((x3, x7), 1)

        # Final processing
        x9 = self.single_conv_block1_3x3(x8)
        x10 = self.single_conv_block2_3x3(x9)
        x_final = F.interpolate(self.final_conv(x10), size=(x_width, x_height),
                                align_corners=False, mode='bilinear')
        return x_final
