import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        residue = x
        B, C, H, W = x.size()

        # Self attention between all the pixels
        x = x.view(B, C, H*W) # (B, C, H, W) -> (B, C, H*W)
        x = x.transpose(-1,-2) # (B, C, H*W) -> (B, H*W, C)
        x = self.attention(x) # (B, H*W, C) -> (B, H*W, C)
        x = x.transpose(-1,-2) # (B, H*W, C) -> (B, C, H*W)
        x = x.view(B, C, H, W) # (B, C, H*W) -> (B, C, H, W)
        x = x + residue
        return x


"""
A residual block is a fundamental building block that helps in training deeper networks 
more effectively. A typical residual block consists of:

- Convolutional Layers: These layers perform the standard convolution operation.

- Normalization Layers: These layers normalize the output of the convolutional 
layers to speed up training and improve stability.
= Activation Functions: Commonly ReLU (Rectified Linear Unit) is used as the activation 
function to introduce non-linearity.

- Shortcut Connection (Skip Connection): This is the key feature of a residual block. 
The input to the block is added to the output of the block after the series of convolutional 
operations. Mathematically, if F(x) represents the transformation applied by 
the convolutional layers and xx is the input, the output of the residual block is x + F(x).

Why Use Residual Blocks? Residual blocks are used for several reasons:

- Mitigating the Vanishing Gradient Problem: In very deep networks, gradients can become 
very small (vanish) during backpropagation, making training difficult. The skip connections 
in residual blocks help gradients flow more easily through the network, alleviating 
this problem.

-Improving Training Efficiency: With the identity shortcut connection, the network can learn 
the identity mapping more easily. This means if some layers are not needed at a particular 
depth, the network can effectively skip them, leading to more efficient training.

- Enabling Deeper Networks: By addressing the vanishing gradient problem, residual blocks 
allow for the training of much deeper networks without a significant increase in training 
error. This enables the creation of more powerful models that can capture more complex 
features.

- Feature Reuse: The shortcut connections allow earlier layers' features to be reused in 
later layers, which can help in preserving and propagating useful information throughout the 
network.

Use in Convolutional Neural Networks

"""

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # This is the residual connection or skip connection, 
        if in_channels == out_channels:
            self.residual_layer  = nn.Identity()
        else:
            self.residual_layer  = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        residue = x
        x = self.groupnorm_1(x) # Still (B, C, H, W)
        x = F.silu(x) # # Still (B, C, H, W)
        x = self.conv_1(x) # Still (B, C, H, W)
        x = self.groupnorm_2(x) # Still (B, C, H, W)
        x = F.silu(x) # Still (B, C, H, W)
        x = self.conv_2(x) # Still (B, C, H, W)

        # Residual connection
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0), # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1), # (B, 4, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_AttentionBlock(512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            
            # Now we need to scale up the image to the original size.
            nn.Upsample(scale_factor=2), # (B, 512, H/8, W/8) -> (B, 512, H/4, W/4)
            nn.Conv2d(512,512, kernel_size=3, padding=1), # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512), # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)

            # Now we need to scale up the image to the original size. 
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            nn.Upsample(scale_factor=2), # (B, 512, H/4, W/4) -> (B, 512, H/2, W/2)
            nn.Conv2d(512,512, kernel_size=3, padding=1), # (B, 512, H/2, W/2) -> (B, 512, H/2, W/2)
            VAE_ResidualBlock(512, 256), # (B, 512, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256), # (B, 256, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256), # (B, 256, H/2, W/2) -> (B, 256, H/2, W/2)

            # Now we need to scale up the image to the original size.
            nn.Upsample(scale_factor=2), # (B, 256, H/2, W/2) -> (B, 256, H, W)
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128), # (B, 256, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128), # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128), # (B, 128, H, W) -> (B, 128, H, W)

            nn.GroupNorm(32, 128), # (B, 128, H, W) -> (B, 128, H, W)
            nn.SiLU(), # (B, 128, H, W) -> (B, 128, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1) # (B, 128, H, W) -> (B, 3, H, W)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H/8, W/8)

        x = x / 0.18215 # Stable diffusion magic number

        for module in self:
            x = module(x)
        
        return x