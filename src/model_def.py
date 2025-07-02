import torch
import torch.nn as nn
import math


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))  # Learnable query vector

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        attn_weights = torch.einsum('bld,d->bl', x, self.query)  # Compute attention scores
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize scores
        return (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # Weighted sum of sequence features


class MultiResolutionTransformer(nn.Module):
    def __init__(self, 
                 daily_input_dim, 
                 hourly_input_dim, 
                 minute5_input_dim, 
                 minute1_input_dim,
                 embed_dim=128, 
                 num_heads=8, 
                 num_layers=2, 
                 output_dim=3, 
                 dropout=0.2):
        super().__init__()

        # Embedding layers for each resolution
        self.daily_embedding = nn.Linear(daily_input_dim, embed_dim)
        self.hourly_embedding = nn.Linear(hourly_input_dim, embed_dim)
        self.minute5_embedding = nn.Linear(minute5_input_dim, embed_dim)
        self.minute1_embedding = nn.Linear(minute1_input_dim, embed_dim)

        # Transformer encoders for each resolution
        self.daily_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.hourly_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.minute5_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.minute1_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        # Attention pooling for each resolution
        self.daily_attention = AttentionPooling(embed_dim)
        self.hourly_attention = AttentionPooling(embed_dim)
        self.minute5_attention = AttentionPooling(embed_dim)
        self.minute1_attention = AttentionPooling(embed_dim)

        # Fully connected layers for output
        self.fc_combined = nn.Linear(embed_dim * 4, embed_dim)  # Combine all resolutions
        self.fc_price = nn.Linear(embed_dim, output_dim)
        self.fc_conf = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.Softplus()  # Confidence must be positive
        )

    def process_resolution(self, x, embedding_layer, transformer, attention_pooling):
        """
        Processes a single resolution: embedding -> transformer -> attention pooling.
        """
        x = embedding_layer(x)  # Apply embedding
        x = x.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        x = transformer(x).permute(1, 0, 2)  # Pass through transformer and back to (batch, seq_len, embed_dim)
        return attention_pooling(x)  # Apply attention pooling

    def forward(self, daily_x, hourly_x, minute5_x, minute1_x):
        """
        Forward pass for the model.
        - daily_x: (batch, seq_len_daily, daily_input_dim)
        - hourly_x: (batch, seq_len_hourly, hourly_input_dim)
        - minute5_x: (batch, seq_len_minute5, minute5_input_dim)
        - minute1_x: (batch, seq_len_minute1, minute1_input_dim)
        """
        # Process each resolution
        daily_feat = self.process_resolution(daily_x, self.daily_embedding, self.daily_transformer, self.daily_attention)
        hourly_feat = self.process_resolution(hourly_x, self.hourly_embedding, self.hourly_transformer, self.hourly_attention)
        minute5_feat = self.process_resolution(minute5_x, self.minute5_embedding, self.minute5_transformer, self.minute5_attention)
        minute1_feat = self.process_resolution(minute1_x, self.minute1_embedding, self.minute1_transformer, self.minute1_attention)

        # Combine all features
        combined_feat = torch.cat([daily_feat, hourly_feat, minute5_feat, minute1_feat], dim=-1)
        combined_feat = self.fc_combined(combined_feat)

        # Predict price and confidence
        price_pred = self.fc_price(combined_feat)
        conf_pred = self.fc_conf(combined_feat)

        return price_pred, conf_pred


class DeltaSenseTransformer(nn.Module):
    """
    A transformer-based classifier for multi-horizon, multi-class crypto price movement prediction.
    Outputs logits for each horizon and class.
    """
    def __init__(self, input_dim, seq_len, n_horizons=6, n_classes=6, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_horizons * n_classes)
        self.n_horizons = n_horizons
        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use last token
        logits = self.head(x)
        logits = logits.view(-1, self.n_horizons, self.n_classes)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
