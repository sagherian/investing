# Active Inference Investing

This project combines active inference agents, hyperbolic discounting, and a PyTorch transformer model to simulate and analyze stock trading strategies.

## Features

- **MarketEnvironment:** Simulates a stock market using real or synthetic price data.
- **HyperbolicActiveInferenceAgent:** Makes buy/hold/sell decisions using active inference and hyperbolic discounting.
- **StockPriceTransformer:** PyTorch transformer model for stock price prediction.
- **Simulation:** Run agents with different `k` values and compare their performance.

## Usage

1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Run the simulation:
    ```
    python src/simulate.py
    ```
3. Explore and visualize in `notebooks/exploratory_analysis.ipynb`.
