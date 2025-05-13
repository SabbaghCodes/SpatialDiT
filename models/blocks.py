import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaptiveLayerNormBlock(nn.Module):
    """
    Learned shift/scale/gate with self-attention and MLP.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaptive_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaptive_norm(cond).chunk(6, dim=1)
        )
        # x: (B, N, H)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class AdaptiveLayerNormBlockV2(nn.Module):
    """
    Learned shift/scale/gate with self-attention and MLP.
    Implements adaptive LayerNorm with zero-initialization strategy for identity-like behavior initially.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaptive_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        
        nn.init.constant_(self.adaptive_norm[1].weight, 0)
        nn.init.constant_(self.adaptive_norm[1].bias, 0)

    def forward(self, x, cond):
        """
        Forward pass of the adaptive layer normalization block.
        :param x: Input tensor.
        :param cond: Condition tensor.
        :return: Output tensor after processing through the adaptive layer normalization block.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaptive_norm(cond).chunk(6, dim=1)
        )
        
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class CustomCrossAttention(nn.Module):
    """
    Q from x, K,V from cond. 
    Different from above 'helper' cross attention for demonstration.
    """
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, cond):
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        B, Nx, H = x.shape
        Nc = cond.shape[1]

        Q = self.q_proj(x).view(B, Nx, self.num_heads, self.head_dim)
        K = self.k_proj(cond).view(B, Nc, self.num_heads, self.head_dim)
        V = self.v_proj(cond).view(B, Nc, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).reshape(B, Nx, H)
        return self.out_proj(out)

class CrossAttentionBlock(nn.Module):
    """
    cross-attn + MLP
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1_x = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm1_cond = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CustomCrossAttention(hidden_size, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, cond):
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)

        x_ = self.norm1_x(x)
        cond_ = self.norm1_cond(cond)
        attn_out = self.cross_attn(x_, cond_)
        x = x + attn_out

        x_2 = self.norm2(x)
        x = x + self.mlp(x_2)
        return x

class InContextConditioningBlock(nn.Module):
    """
    Concatenate x and cond in sequence dim -> self-attn -> slice out x.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, cond):
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        
        concatenated = torch.cat([x, cond], dim=1)
        concatenated_norm = self.norm1(concatenated)
        attn_out, _ = self.attn(concatenated_norm, concatenated_norm, concatenated_norm)
        
        x_out = attn_out[:, : x.size(1), :]
        x = x + x_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x
