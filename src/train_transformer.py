import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.transformer_model import StockPriceTransformer

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_transformer(price_series, seq_length=20, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPriceTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data = price_series.values.reshape(-1, 1)
    xs, ys = create_sequences(data, seq_length)
    xs = torch.tensor(xs, dtype=torch.float32).transpose(0, 1).to(device)  # (seq_len, batch, input_dim)
    ys = torch.tensor(ys, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(xs).squeeze(-1)[-1]  # predict last in sequence
        loss = criterion(output, ys)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model
