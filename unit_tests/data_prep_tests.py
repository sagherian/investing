import pandas as pd
import sys
import os
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../src"))
from data_prep_utils import *
import pytest


@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = {
        "Close": np.random.rand(100) * 100 + 100,
        "High": np.random.rand(100) * 100 + 110,
        "Low": np.random.rand(100) * 100 + 90,
        "Open": np.random.rand(100) * 100 + 95,
        "Volume": np.random.randint(1e6, 1e7, size=100),
    }
    return pd.DataFrame(data, index=dates)

def test_compute_rsi(sample_data):
    df = compute_rsi(sample_data.copy(), column="Close")
    assert "RSI" in df.columns
    assert df["RSI"].isna().sum() == 14  # First 14 values should be NaN
    assert df["RSI"].iloc[15:].notna().all()  # After 14 values, RSI should be computed

def test_compute_ema(sample_data):
    df = compute_ema(sample_data.copy(), column="Close", windows=[12, 26])
    assert "EMA_12" in df.columns
    assert "EMA_26" in df.columns
    assert df["EMA_12"].isna().sum() == 11  # First 12 values should be NaN
    assert df["EMA_26"].isna().sum() == 25  # First 26 values should be NaN

def test_compute_macd(sample_data):
    df = compute_macd(sample_data.copy(), column="Close")
    assert "MACD" in df.columns
    assert "Signal" in df.columns
    assert df["MACD"].isna().sum() == 25  # Longest window for MACD

def test_compute_bollinger_bands(sample_data):
    df = compute_bollinger_bands(sample_data.copy(), column="Close")
    assert "BB_upper" in df.columns
    assert "BB_lower" in df.columns
    assert df["BB_upper"].isna().sum() == 19  # Window for Bollinger Bands
    assert df["BB_lower"].isna().sum() == 19

def test_compute_obv(sample_data):
    df = compute_obv(sample_data.copy(), column="Close", volume_column="Volume")
    assert "OBV" in df.columns
    assert df["OBV"].isna().sum() == 0  # OBV should not have NaN

def test_compute_atr(sample_data):
    df = compute_atr(sample_data.copy(), high_col="High", low_col="Low", close_col="Close")
    assert "ATR" in df.columns
    assert df["ATR"].isna().sum() == 13  # ATR window is 14

def test_enrich_with_technical_indicators(sample_data):
    sample_data.columns = pd.MultiIndex.from_tuples(
        [(col, "AAPL") for col in sample_data.columns], names=["Price", "Ticker"]
    )
    df = enrich_with_technical_indicators(sample_data)
    assert all(col.startswith("AAPL_") for col in df.columns)
    assert "AAPL_RSI" in df.columns
    assert "AAPL_MACD" in df.columns

def test_download_and_enrich_data(mocker):
    mocker.patch("yfinance.download", return_value=sample_data())
    tickers = {"Apple": "AAPL", "Microsoft": "MSFT"}
    context_data = download_and_enrich_data(tickers)
    assert "Apple" in context_data
    assert "Microsoft" in context_data
    assert "AAPL_RSI" in context_data["Apple"].columns
    assert "MSFT_RSI" in context_data["Microsoft"].columns
