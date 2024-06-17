import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.linear_2 = nn.Linear(4 * embedding_size, 4 * embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = F.silu(self.linear_1(x))
        x = self.linear_2(x)

        # x: (1, 1280)
        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280) -> None:
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residue_layer = nn.Identity()
        else:
            self.residue_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (B, C, H, W)
        # time: (1, 1280)

        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(self.conv_feature(feature))

        time = F.silu(time) # Is the order of the activation function correct?
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residue_layer(residue)
    
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, d_context=768) -> None:
        super().__init__()
        channels = n_heads * embed_size
        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_gelu_1 = nn.Linear(channels * 4, channels * 2)
        self.linear_gelu_2 = nn.Linear(channels * 4, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, F, H, W)
        # context: (B, T, C)

        B, F, H, W = x.shape

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        x = x.view(B, F, H * W).transpose(-1, -2) # (B, F, H * W) -> (B, H * W, F)

        # Normalization + self-attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short
        residue_short = x

        # Normalization + cross-attention with skip connection 
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residue_short
        residue_short = x

        # Normalization + feedforward with GeGLU and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_gelu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_gelu_2(x)
        x = x + residue_short

        # (B, H * W, F) -> (B, F, H, W)
        x = x.transpose(-1, -2).view(B, F, H, W)

        return self.conv_output(x) + residue_long

class UpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, H * 2, W * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    """
    A Sequential container with a switch method to allow for dynamic changes in the forward pass
    """
    def forward(self, x: torch.Tensor, context: torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            # (B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (B, 320, H/8, W/8) -> (B, 320, H/16, W/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (B, 640, H/16, W/16) -> (B, 640, H/32, W/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (B, 1280, H/32, W/32) -> (B, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            # (B, 2560, H/64, W/64) -> (B, 1280, H/32, W/32)
            # it is double the input compared to the encoders because there are skip connection on the residual blocks. 
            #The original UNET has skip connections that go from the encoder to the decoder.
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class UNET_output(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnom = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 320, H/8, W/8) -> 
        x = F.silu(self.groupnom(x))
        x = self.conv(x)

        # x: (B, 4, H/8, W/8)
        return x

class Difussion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_output(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (B, 4, H/8, W/8) as this is the output of the encoder
        # context: (B, T, C) as this is the output of the clip model. C is the embedding size whics is 768
        # time: (1, 320)

        time = self.time_embedding(time) # (1, 320) -> (1, 1280)

        #(B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
        output = self.unet(latent, context, time)

        # (B, 320, H/8, W/8) -> (B, 4, H/8, W/8)
        output = self.final(output)

        # output: (B, 4, H/8, W/8)
        return output