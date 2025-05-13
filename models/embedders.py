import math
import torch
import torch.nn as nn

class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10000):
        device = t.device
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=device) / half)
        t = t.view(-1)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_emb = self.sinusoidal_embedding(t, self.mlp[0].in_features)
        return self.mlp(t_emb)
