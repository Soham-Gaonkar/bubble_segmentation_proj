# model/attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Example Attention Gate (you can find implementations online)
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g = self.W_g(g)
        x = self.W_x(x)

        # Resize g to match the size of x
        g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g + x)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):  # Adjusted in_channels
        super().__init__()
        # Example simplified structure, replace with actual U-Net architecture
        self.encoder1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.attention1 = AttentionGate(128, 64)

        self.decoder1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # Example, adjust sizes
        self.decoder2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(self.pool(e1)))

        # Attention
        a1 = self.attention1(e2, e1) # Apply attention

        # Decoder
        d1 = F.relu(self.decoder1(a1))
        output = torch.sigmoid(self.decoder2(d1))  # Sigmoid for binary segmentation

        return output

# Example Usage (for testing)
if __name__ == '__main__':
    model = AttentionUNet(in_channels=1, num_classes=1)
    input_tensor = torch.randn(1, 1, 256, 256)  # Example input: batch size 1, 1 channel, 256x256
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected: torch.Size([1, 1, 256, 256])