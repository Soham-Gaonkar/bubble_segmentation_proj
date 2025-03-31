# model/attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building Blocks ---

class ConvBlock(nn.Module):
    """Standard Double Convolution Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # Bias False with BN
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AttentionGate(nn.Module):
    """Attention Gate (Grid Attention) for Attention U-Net"""
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Number of channels in the gating signal (from deeper layer).
            F_l (int): Number of channels in the input signal (from skip connection).
            F_int (int): Number of channels in the intermediate layer.
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from the deeper layer.
            x: Input signal from the skip connection.
        Returns:
            Attention-weighted input signal (x * attention coefficients).
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g1 to match the spatial dimensions of x1 if necessary
        if g1.shape[-2:] != x1.shape[-2:]:
             g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi) # Attention coefficients (alpha)

        return x * psi # Apply attention coefficients to the original input signal x

class UpConvBlock(nn.Module):
    """
    Up-Convolution Block: Upsample(ConvTranspose2d) -> Concatenate -> DoubleConv
    Takes channels from below (in_ch) and skip connection (skip_ch).
    Outputs skip_ch channels.
    """
    def __init__(self, in_ch, skip_ch):
        super().__init__()
        # Upsample layer reduces channels by half (common practice)
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # Conv block takes concatenated channels (skip_ch from skip + in_ch // 2 from upsampled)
        # The output channels of this block should match the skip connection channels
        self.conv = ConvBlock(skip_ch + in_ch // 2, skip_ch)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from the previous layer in the decoder path (to be upsampled).
            x2: Skip connection input (potentially attention-weighted) from the encoder.
        """
        x1 = self.up(x1) # Upsample the feature map from the deeper layer

        # Pad x1 (upsampled) to match the spatial dimensions of x2 (skip connection)
        # This handles cases where input size wasn't perfectly divisible by 2^N
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- Attention U-Net Model ---
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512, 1024]):
        """
        Standard Attention U-Net Architecture.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes (e.g., 1 for binary segmentation).
            features (list): List of feature channels at each level of the U-Net.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, features[0]) # Out: 64
        self.enc2 = ConvBlock(features[0], features[1]) # Out: 128
        self.enc3 = ConvBlock(features[1], features[2]) # Out: 256
        self.enc4 = ConvBlock(features[2], features[3]) # Out: 512

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(features[3], features[4]) # Out: 1024

        # --- Decoder ---
        # Attention Gates (F_g from below, F_l from skip connection)
        self.att4 = AttentionGate(F_g=features[4], F_l=features[3], F_int=features[3] // 2)
        self.att3 = AttentionGate(F_g=features[3], F_l=features[2], F_int=features[2] // 2)
        self.att2 = AttentionGate(F_g=features[2], F_l=features[1], F_int=features[1] // 2)
        self.att1 = AttentionGate(F_g=features[1], F_l=features[0], F_int=features[0] // 2)

        # Upsampling blocks (in_ch from below, skip_ch from corresponding encoder layer)
        self.up4 = UpConvBlock(in_ch=features[4], skip_ch=features[3]) # Out: features[3] (512)
        self.up3 = UpConvBlock(in_ch=features[3], skip_ch=features[2]) # Out: features[2] (256)
        self.up2 = UpConvBlock(in_ch=features[2], skip_ch=features[1]) # Out: features[1] (128)
        self.up1 = UpConvBlock(in_ch=features[1], skip_ch=features[0]) # Out: features[0] (64)

        # --- Output Layer ---
        self.out_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        e1 = self.enc1(x)                # Ch: 64,  Size: H, W
        p1 = self.pool(e1)               # Size: H/2, W/2
        e2 = self.enc2(p1)               # Ch: 128, Size: H/2, W/2
        p2 = self.pool(e2)               # Size: H/4, W/4
        e3 = self.enc3(p2)               # Ch: 256, Size: H/4, W/4
        p3 = self.pool(e3)               # Size: H/8, W/8
        e4 = self.enc4(p3)               # Ch: 512, Size: H/8, W/8
        p4 = self.pool(e4)               # Size: H/16, W/16

        # --- Bottleneck ---
        b = self.bottleneck(p4)          # Ch: 1024, Size: H/16, W/16

        # --- Decoder Path ---
        # Level 4
        a4 = self.att4(g=b, x=e4)         # Apply attention to e4 using b as gating signal
        d4 = self.up4(x1=b, x2=a4)        # Upsample b, concat with attended e4. Out Ch: 512, Size: H/8, W/8

        # Level 3
        a3 = self.att3(g=d4, x=e3)        # Apply attention to e3 using d4 as gating signal
        d3 = self.up3(x1=d4, x2=a3)       # Upsample d4, concat with attended e3. Out Ch: 256, Size: H/4, W/4

        # Level 2
        a2 = self.att2(g=d3, x=e2)        # Apply attention to e2 using d3 as gating signal
        d2 = self.up2(x1=d3, x2=a2)       # Upsample d3, concat with attended e2. Out Ch: 128, Size: H/2, W/2

        # Level 1
        a1 = self.att1(g=d2, x=e1)        # Apply attention to e1 using d2 as gating signal
        d1 = self.up1(x1=d2, x2=a1)       # Upsample d2, concat with attended e1. Out Ch: 64, Size: H, W

        # --- Output ---
        logits = self.out_conv(d1)       # Out Ch: num_classes, Size: H, W

        # Return logits; sigmoid is handled by the loss function (DiceFocalLoss)
        return logits