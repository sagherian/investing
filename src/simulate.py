import yfinance as yf
import numpy as np
import torch
from src.environment import MarketEnvironment
from src.agent import HyperbolicActiveInferenceAgent
from src.train_transformer import train_transformer

def predict_prices(model, price_series, seq_length=20):
    model.eval()
    data = price_series.values.reshape(-1, 1)
    preds = []
    for i in range(len(data)):
        if i < seq_length:
            preds.append(data[i][0])
        else:
            seq = torch.tensor(data[i-seq_length:i], dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                pred = model(seq).squeeze(-1)[-1].item()
            preds.append(pred)
    return np.array(preds)

def run_simulation(price_series, k, transformer_model, seq_length=20):
    env = MarketEnvironment(price_series)
    agent = HyperbolicActiveInferenceAgent(k=k)
    predicted_prices = predict_prices(transformer_model, price_series, seq_length)
    state = env.reset()
    done = False
    while not done:
        action = agent.plan(env, predicted_prices)
        state, _, done = env.step(action)
    history = env.get_history()
    return history

if __name__ == "__main__":
    # Download data
    df = yf.download('COIN', period='1y', progress=False)
    price_series = df['Close'].dropna().reset_index(drop=True)

    # Train transformer
    transformer = train_transformer(price_series, seq_length=20, epochs=5)

    # Simulate for different k values
    k_values = [0.01, 0.1, 0.5, 1.0]
    results = {}
    for k in k_values:
        history = run_simulation(price_series, k, transformer, seq_length=20)
        results[k] = history

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for k, history in results.items():
        plt.plot(history['t'], history['portfolio_value'], label=f'k={k}')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time for Different Impulsivity (k) Values')
    plt.legend()
    plt.tight_layout()
    plt.show()
