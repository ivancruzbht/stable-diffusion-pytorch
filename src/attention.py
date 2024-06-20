import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.size()
        interm_shape = (B, T, self.n_heads, self.d_head)

        q,k,v = self.in_proj(x).chunk(3, dim=-1) # (B, C, T) -> (B, T, 3*C) -> 3*(B, T, C)

        # (B, T, C) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.view(*interm_shape).transpose(1,2) 
        k = k.view(*interm_shape).transpose(1,2) 
        v = v.view(*interm_shape).transpose(1,2) 
        
        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        attention = q @ k.transpose(-1,-2) / math.sqrt(self.d_head)

        if causal_mask:
            mask = torch.ones_like(attention, dtype=torch.bool).triu(1)
            attention.masked_fill_(mask, float('-inf'))
            
        attention = F.softmax(attention, dim=-1)
        
        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        output = attention @ v
        # (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, C)
        output = output.transpose(1,2).contiguous().view(B, T, C)
        # (B, T, C) -> (B, T, C)
        output = self.out_proj(output)

        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, cross_size: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(embed_size, embed_size, bias=in_proj_bias)
        self.k_proj = nn.Linear(cross_size, embed_size, bias=in_proj_bias)
        self.v_proj = nn.Linear(cross_size, embed_size, bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=out_proj_bias)
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, T_Q, C_Q)
        # y: (B, T_KV, C_KV) = (B, 77, 768)

        B, T, C = x.size()
        interm_shape = (B, -1, self.n_heads, self.head_size) # T=-1 because we don't know the size of the sequence (T_Q != T_KV)
                
        q = self.q_proj(x).view(interm_shape).transpose(1,2)
        k = self.k_proj(y).view(interm_shape).transpose(1,2) 
        v = self.v_proj(y).view(interm_shape).transpose(1,2)

        attention = q @ k.transpose(-1,-2) / math.sqrt(self.head_size)
        attention = F.softmax(attention, dim=-1)

        output = attention @ v
        output = output.transpose(1,2).contiguous().view(B, T, C)
        output = self.out_proj(output)

        return output


    