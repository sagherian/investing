import torch
import torch.nn as nn
import torch.nn.functional as F

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
