import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
This is just the text CLIP encoder. The CLIP text encoder is based on the Transformer architecture.
"""

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, context_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.zeros(context_size, embed_size))

    def forward(self, tokens: torch.LongTensor):
        # (B, T) -> (B, T, C)
        x = self.token_embedding(tokens) # (B, T) -> (B, T, C)
        x = x + self.position_embedding # (B, T, C) + (T, C) -> (B, T, C) via broadcasting

        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_head: int, embed_size: int):
        super().__init__()

        self.layernorm_1  = nn.LayerNorm(embed_size)
        self.attention  = SelfAttention(n_head, embed_size)
        self.layernorm_2  = nn.LayerNorm(embed_size)
        self.linear_1  = nn.Linear(embed_size, 4 * embed_size)
        self.linear_2  = nn.Linear(4 * embed_size, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residue = x

        # Self-Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x = x + residue

        # Feed Forward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = F.gelu(x) # original code uses x = x * torch.sigmoid(1.702 * x), which is QuickGELU activation function
        x = self.linear_2(x)
        x = x + residue

        return x # (B, T, C)

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) # Vocab size, Embed size, Context size
        self.layers = nn.ModuleList(
            [TransformerBlock(n_head=12, embed_size=768) for i in range(12)]
        )
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        x = self.embedding(tokens) # (B, T) -> (B, T, C) or (B, T, 768)
        for layer in self.layers:
            x = layer(x)
        
        output = self.layernorm(x) # (B, T, 768) -> (B, T, 768)
        return output
