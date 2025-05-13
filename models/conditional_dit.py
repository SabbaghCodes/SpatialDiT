import torch
import torch.nn as nn

from models.embedders import TimestepEmbedder
from models.blocks import (
    AdaptiveLayerNormBlock,
    AdaptiveLayerNormBlockV2,
    CrossAttentionBlock,
    InContextConditioningBlock
)

class ConditionalDiT(nn.Module):
    """
    DiT model that can switch block_type:
      - "adaLN" -> AdaptiveLayerNormBlock
      - "cross_attention" -> CrossAttentionBlock
      - "in_context" -> InContextConditioningBlock

    Also includes a small cond_mapper to handle dimension mismatch
    between cond.size(1) and hidden_size if needed.
    Also includes patch_embedding -> hidden_size for x.
    """
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_layers,
        num_heads,
        mlp_ratio=4.0,
        block_type="adaLN",
        cond_dim=None,  
        add_variance_in_reverse=False
    ):
        """
        add_variance_in_reverse: whether to add noise in reverse diffusion steps
        """
        super().__init__()
        self.patch_embedding = nn.Linear(input_dim, hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.add_variance_in_reverse = add_variance_in_reverse

        if block_type == "adaLN":
            Block = AdaptiveLayerNormBlockV2
        elif block_type == "cross_attention":
            Block = CrossAttentionBlock
        elif block_type == "in_context":
            Block = InContextConditioningBlock
        else:
            raise ValueError("Unsupported block type")

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, input_dim)

        if cond_dim is not None and cond_dim != hidden_size:
            self.cond_mapper = nn.Linear(cond_dim, hidden_size)
        else:
            self.cond_mapper = None

    def forward(self, x_gene, t, cond):
        """
        x_gene: (B, input_dim)
        cond:   (B, cond_dim)
        """
        t_emb = self.time_embedder(t)  
        
        x_emb = self.patch_embedding(x_gene)  
        
        if self.cond_mapper is not None:
            cond = self.cond_mapper(cond) 
        c = t_emb + cond  

        x_emb = x_emb.unsqueeze(1)  
        for block in self.blocks:
            x_emb = block(x_emb, c)

        out = self.output_layer(x_emb.squeeze(1))
        return out
