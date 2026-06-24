"""Strip-Level Self-Attention (mathematisch GNN auf voll-verbundenem Graph)."""
from __future__ import annotations

import torch
import torch.nn as nn


class StripModel(nn.Module):
    def __init__(self, n_strip_feats=6, n_global_feats=3, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.strip_encoder = nn.Sequential(
            nn.Linear(n_strip_feats, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.global_proj = nn.Linear(n_global_feats, d_model)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, strip_feats, mask, global_feats):
        h = self.strip_encoder(strip_feats)
        h = self.transformer(h, src_key_padding_mask=~mask)

        m = mask.unsqueeze(-1).float()
        mean_h = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        max_h  = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(1)
        a      = self.attn_pool(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        attn_h = (h * torch.softmax(a, dim=1).unsqueeze(-1)).sum(1)

        z = torch.cat([mean_h, max_h, attn_h, self.global_proj(global_feats)], dim=-1)
        return self.head(z).squeeze(-1)
