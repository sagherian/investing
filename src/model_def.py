import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, embed_dim))  # Max sequence length = 500
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fc_price = nn.Linear(embed_dim, 4)  # Predict prices
        self.fc_conf = nn.Linear(embed_dim, 4)  # Predict confidence values

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer(x, x)  # Encoder-only architecture
        x = x.mean(dim=1)  # Pooling over the sequence
        price_preds = self.fc_price(x)  # Price predictions
        conf_preds = F.softplus(self.fc_conf(x))  # Confidence values, constrained to positive
        return price_preds, conf_preds
