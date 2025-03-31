# ================================================
# File: code_files/model/resnet18.py
# ================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# --- Building Blocks ---

class ConvBlock(nn.Module):
    """Standard Double Convolution Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpConvResnet(nn.Module):
    """Upsampling block for ResNet-UNet"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Use ConvTranspose2d for upsampling
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # Conv block takes concatenated channels (skip_ch + in_ch // 2)
        self.conv = ConvBlock(skip_ch + in_ch // 2, out_ch)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from the previous layer in the decoder path (to be upsampled).
            x2: Skip connection input from the corresponding encoder layer.
        """
        x1 = self.up(x1)
        # Pad x1 to match x2 size if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- ResNet18 U-Net Model ---

class ResNet18CNN(nn.Module):
    """U-Net architecture with a ResNet18 encoder."""
    def __init__(self, in_channels=3, num_classes=1, pretrained=False):
        super().__init__()

        # --- Encoder (ResNet18 Backbone) ---
        if pretrained:
            # This block will only run if pretrained=True is explicitly passed during instantiation
            print("Loading pretrained ResNet18 weights.")
            weights = ResNet18_Weights.DEFAULT
        else:
            print("Initializing ResNet18 backbone weights from scratch (or using base random init).")
            weights = None

        # Load the base ResNet18 architecture structure
        resnet_model = resnet18(weights=weights)

        # Modify first layer if input channels are not 3
        if in_channels != 3:
            print(f"Adapting ResNet18 first conv layer from 3 to {in_channels} channels.")
            original_conv1 = resnet_model.conv1
            self.encoder_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels,
                                            kernel_size=original_conv1.kernel_size,
                                            stride=original_conv1.stride,
                                            padding=original_conv1.padding,
                                            bias=original_conv1.bias is not None)
            # --- Weight initialization heuristic if needed when not using ImageNet weights ---
            # If pretrained=True was used above, weights would be initialized by averaging.
            # If pretrained=False, the new layer has default random initialization.
            # You *could* uncomment the averaging below even if pretrained=False,
            # but it wouldn't make much sense as the original weights aren't from ImageNet.
            # if pretrained and original_conv1.bias is not None: # Check if bias exists
            #     self.encoder_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            #     if original_conv1.bias is not None:
            #            self.encoder_conv1.bias.data = original_conv1.bias.data

        else: # if in_channels == 3
            self.encoder_conv1 = resnet_model.conv1 # Use original conv1

        # Assign other ResNet layers to the encoder parts
        self.encoder_bn1 = resnet_model.bn1
        self.encoder_relu = resnet_model.relu
        self.encoder_maxpool = resnet_model.maxpool
        self.encoder_layer1 = resnet_model.layer1 # Output channels: 64
        self.encoder_layer2 = resnet_model.layer2 # Output channels: 128
        self.encoder_layer3 = resnet_model.layer3 # Output channels: 256
        self.encoder_layer4 = resnet_model.layer4 # Output channels: 512

        # --- Decoder ---
        # These blocks progressively upsample and combine with skip connections
        self.up4 = UpConvResnet(in_ch=512, skip_ch=256, out_ch=256)
        self.up3 = UpConvResnet(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = UpConvResnet(in_ch=128, skip_ch=64, out_ch=64)

        # Convolution block after concatenating the highest-resolution skip connection
        # Input channels = 64 (from up2 output) + 64 (from skip connection x0)
        self.up1_conv = ConvBlock(in_channels=64 + 64, out_channels=64)

        # Final 1x1 convolution to map features to the specified number of output classes
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        # Initial block processing
        x0 = self.encoder_conv1(x)
        x0 = self.encoder_bn1(x0)
        x0 = self.encoder_relu(x0) # Features after initial block (used as skip connection later)
                                   # Output channels: 64, Size: H/2, W/2

        p0 = self.encoder_maxpool(x0) # Apply max pooling. Size: H/4, W/4

        # Pass through ResNet layers, saving outputs for skip connections
        e1 = self.encoder_layer1(p0) # Output channels: 64, Size: H/4, W/4
        e2 = self.encoder_layer2(e1) # Output channels: 128, Size: H/8, W/8
        e3 = self.encoder_layer3(e2) # Output channels: 256, Size: H/16, W/16
        e4 = self.encoder_layer4(e3) # Output channels: 512, Size: H/32, W/32 (Deepest features)

        # --- Decoder ---
        # Upsample and combine features layer by layer
        d4 = self.up4(x1=e4, x2=e3) # Upsample e4 (deepest), concat w/ e3. Out: 256ch, H/16, W/16
        d3 = self.up3(x1=d4, x2=e2) # Upsample d4, concat w/ e2. Out: 128ch, H/8, W/8
        d2 = self.up2(x1=d3, x2=e1) # Upsample d3, concat w/ e1. Out: 64ch, H/4, W/4

        # Upsample the result from the previous decoder stage (d2)
        # Use bilinear interpolation for upsampling
        d2_upsampled = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False) # Size: H/2, W/2

        # Pad d2_upsampled to match the spatial size of x0 if necessary
        diffY = x0.size()[2] - d2_upsampled.size()[2]
        diffX = x0.size()[3] - d2_upsampled.size()[3]
        if diffY > 0 or diffX > 0: # Only pad if needed
             d2_upsampled = F.pad(d2_upsampled, [diffX // 2, diffX - diffX // 2,
                                                 diffY // 2, diffY - diffY // 2])

        # Concatenate the upsampled features (d2_upsampled) with the early encoder features (x0)
        d1 = torch.cat([x0, d2_upsampled], dim=1) # Concat 64ch + 64ch -> 128ch

        # Process the concatenated features through the final decoder conv block
        d1 = self.up1_conv(d1) # Out: 64ch, H/2, W/2

        # Final upsampling to the original input image size
        d0 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False) # Size H, W

        # Apply the final 1x1 convolution to get the output logits for each class
        logits = self.out_conv(d0) # Out: num_classes channels, Size: H, W

        # Return logits; activation (like sigmoid for binary) should be applied in the loss or post-processing
        return logits

# Example Usage Block (useful for testing the model file independently)
if __name__ == '__main__':
    # Test grayscale, not using ImageNet weights
    model_gray = ResNet18CNN(in_channels=1, num_classes=1, pretrained=False)
    input_gray = torch.randn(2, 1, 256, 256) # Batch=2, Channels=1, Size=256x256
    output_gray = model_gray(input_gray)
    print("ResNet18CNN (Gray Input, No Pretrained ImageNet)")
    print(f"Input shape: {input_gray.shape}")
    print(f"Output shape: {output_gray.shape}") # Should be [2, 1, 256, 256]
    num_params_gray = sum(p.numel() for p in model_gray.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params_gray:,}")

    print("-" * 20)

    # Test RGB, not using ImageNet weights
    model_rgb = ResNet18CNN(in_channels=3, num_classes=5, pretrained=False)
    input_rgb = torch.randn(1, 3, 224, 224) # Batch=1, Channels=3, Size=224x224
    output_rgb = model_rgb(input_rgb)
    print("ResNet18CNN (RGB Input, No Pretrained ImageNet)")
    print(f"Input shape: {input_rgb.shape}")
    print(f"Output shape: {output_rgb.shape}") # Should be [1, 5, 224, 224]
    num_params_rgb = sum(p.numel() for p in model_rgb.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params_rgb:,}")