import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class MultiResolutionTransformer(nn.Module):
    def __init__(self, 
                 daily_input_dim, 
                 minute_input_dim, 
                 embed_dim=128, 
                 num_heads=8, 
                 num_layers=2, 
                 output_dim=3, 
                 dropout=0.2):
        super().__init__()
        # Daily transformer encoder
        self.daily_embedding = nn.Linear(daily_input_dim, embed_dim)
        self.daily_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        # Minute transformer encoder
        self.minute_embedding = nn.Linear(minute_input_dim, embed_dim)
        self.minute_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        # Combine both resolutions
        self.combined_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.Softplus()  # Confidence must be positive
        )

    def forward(self, daily_x, minute_x):
        # daily_x: (batch, seq_len_daily, daily_input_dim)
        # minute_x: (batch, seq_len_minute, minute_input_dim)
        daily_emb = self.daily_embedding(daily_x)
        daily_emb = daily_emb.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        daily_out = self.daily_transformer(daily_emb)
        daily_feat = daily_out[-1]  # Use last token

        minute_emb = self.minute_embedding(minute_x)
        minute_emb = minute_emb.permute(1, 0, 2)
        minute_out = self.minute_transformer(minute_emb)
        minute_feat = minute_out[-1]

        combined = torch.cat([daily_feat, minute_feat], dim=-1)
        combined = self.combined_fc(combined)
        price_pred = self.head(combined)
        conf_pred = self.conf_head(combined)
        return price_pred, conf_pred

class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim=4, seq_length=500, positional_buffer=16, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        
        self.seq_length = seq_length
        self.max_positional_length = seq_length + positional_buffer  # Buffer for longer sequences
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_positional_length, embed_dim))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.fc_price = nn.Linear(embed_dim, output_dim)
        self.fc_conf = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        seq_len = x.size(1)
        
        # Ensure the input sequence length does not exceed the maximum positional length
        if seq_len > self.max_positional_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds the configured maximum ({self.max_positional_length}).")
        
        # Add embeddings and positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        
        # Pass through the encoder
        x = self.encoder(x)
        
        # Pooling over the sequence
        x = x.mean(dim=1)
        
        # Output layers
        price_preds = self.fc_price(x)
        # conf_preds = F.softplus(self.fc_conf(x))
        conf_preds = torch.sigmoid(self.fc_conf(x))  # Now between 0 and 1

        return price_preds, conf_preds
