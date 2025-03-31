# model/deeplabv3plus.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# --- ASPP Module ---

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:] # Get H, W
        # Apply pooling and conv
        x = super().forward(x)
        # Upsample back to original H, W
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    Args:
        in_channels (int): Number of channels in the input feature map.
        atrous_rates (list): List of dilation rates for the parallel branches.
        out_channels (int): Number of channels produced by the ASPP module.
    """
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = []
        # 1x1 Convolution branch
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous Convolution branches
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Image Pooling branch
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Final convolution to fuse features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) # Added dropout as commonly used in DeepLab
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1) # Concatenate along channel dimension
        return self.project(res)


# --- DeepLabV3+ Model ---

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture with a ResNet50 backbone.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        num_classes (int): Number of output segmentation classes.
        output_stride (int): Desired output stride of the backbone (8 or 16).
                             Affects dilation rates in backbone and ASPP.
        pretrained (bool): If True, load weights pretrained on ImageNet.
                           Default is False.
    """
    def __init__(self, in_channels=3, num_classes=1, output_stride=16, pretrained=False):
        super().__init__()
        if output_stride not in [8, 16]:
            raise ValueError("output_stride must be 8 or 16")

        # --- Backbone (ResNet50) ---
        if pretrained:
            print("Loading pretrained ResNet50 weights for DeepLabV3+.")
            weights = ResNet50_Weights.DEFAULT
        else:
            print("Initializing ResNet50 backbone from scratch for DeepLabV3+.")
            weights = None # Will use random initialization

        resnet_model = resnet50(weights=weights,
                                replace_stride_with_dilation=[False, output_stride==8, True])

        # Modify first layer if input channels are not 3
        if in_channels != 3:
            print(f"Adapting ResNet50 first conv layer from 3 to {in_channels} channels.")
            original_conv1 = resnet_model.conv1
            self.backbone_conv1 = nn.Conv2d(in_channels, original_conv1.out_channels,
                                            kernel_size=original_conv1.kernel_size,
                                            stride=original_conv1.stride,
                                            padding=original_conv1.padding,
                                            bias=original_conv1.bias is not None)
            # Weight initialization logic (optional if not pretrained)
            # if pretrained: # Example: average weights if pretrained
            #     with torch.no_grad():
            #         self.backbone_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            #         if original_conv1.bias is not None:
            #                self.backbone_conv1.bias.data = original_conv1.bias.data
        else:
            self.backbone_conv1 = resnet_model.conv1

        # Extract ResNet layers, excluding fc and avgpool
        self.backbone_bn1 = resnet_model.bn1
        self.backbone_relu = resnet_model.relu
        self.backbone_maxpool = resnet_model.maxpool
        self.backbone_layer1 = resnet_model.layer1 # Output stride 4, channels 256 (after 1x1)
        self.backbone_layer2 = resnet_model.layer2 # Output stride 8, channels 512
        self.backbone_layer3 = resnet_model.layer3 # Output stride 8 or 16, channels 1024
        self.backbone_layer4 = resnet_model.layer4 # Output stride 16 or 16, channels 2048

        # --- ASPP ---
        # Determine ASPP dilation rates based on output stride
        if output_stride == 16:
            aspp_rates = [6, 12, 18]
            high_level_channels = 2048 # Output channels of ResNet layer4
        elif output_stride == 8:
            aspp_rates = [12, 24, 36]
            high_level_channels = 2048 # Output channels of ResNet layer4 (still)
        else: # Should not happen due to initial check
             raise NotImplementedError

        self.aspp = ASPP(in_channels=high_level_channels, atrous_rates=aspp_rates, out_channels=256)

        # --- Decoder ---
        low_level_channels = 256 # Output channels of ResNet layer1

        # 1x1 convolution to reduce channels of low-level features
        self.decoder_conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolutions to fuse features after concatenation
        self.decoder_conv_fuse = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), # ASPP channels + reduced low-level channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Common practice in DeepLab decoders
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # --- Classifier ---
        # Final 1x1 convolution to get class scores
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:] # H, W

        # --- Backbone ---
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x) # Stride 4 feature map start point

        low_level_features = self.backbone_layer1(x) # Stride 4, 256 channels
        x = self.backbone_layer2(low_level_features) # Stride 8, 512 channels
        x = self.backbone_layer3(x)                  # Stride 8 or 16, 1024 channels
        high_level_features = self.backbone_layer4(x) # Stride 16, 2048 channels

        # --- ASPP ---
        x = self.aspp(high_level_features) # Output shape (B, 256, H/16, W/16)

        # --- Decoder ---
        # Upsample ASPP output to match low-level feature size (stride 4)
        x = F.interpolate(x, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False)

        # Process low-level features
        low_level_features = self.decoder_conv_low(low_level_features) # (B, 48, H/4, W/4)

        # Concatenate features
        x = torch.cat((x, low_level_features), dim=1) # (B, 256 + 48, H/4, W/4)

        # Fuse features
        x = self.decoder_conv_fuse(x) # (B, 256, H/4, W/4)

        # Upsample to original input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        # --- Classifier ---
        logits = self.classifier(x) # (B, num_classes, H, W)

        return logits