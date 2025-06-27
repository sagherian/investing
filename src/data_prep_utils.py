### Functions for Data Prep ###
import pandas as pd
import numpy as np
import yfinance as yf

# Download historical data for context tickers and apply technical indicators
def download_and_enrich_data(tickers: dict, period = 'max', interval='1d'):
    context_data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df = enrich_with_technical_indicators(df)
        context_data[name] = df
    return context_data

def enrich_with_technical_indicators(df):
    ticker = df.columns[0][1]
    df = compute_ema(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_obv(df)
    df = compute_atr(df)
    df = df.dropna()
    df.columns = [f"{ticker}_{col[0]}" for col in df.columns]

    return df

### Individual Technical Indicators ###
# Relative Strength Index (RSI)
def compute_rsi(df, column='Close', window=14):
    delta = df[column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

# Exponential Moving Average (EMA)
def compute_ema(df, column='Close', windows=[12, 26]):
    for window in windows:
        df[f'EMA_{window}'] = df[column].ewm(span=window, adjust=False).mean()
    return df

# Moving Average Convergence Divergence (MACD)
def compute_macd(df, column='Close', short_window=12, long_window=26, signal_window=9):
    short_ema = df[column].ewm(span=short_window, adjust=False).mean()
    long_ema = df[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    df['MACD'] = macd
    df['Signal'] = signal
    return df

# Bollinger Bands
def compute_bollinger_bands(df, column='Close', window=20):
    ma = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()
    df['BB_upper'] = ma + 2 * std
    df['BB_lower'] = ma - 2 * std
    return df

# On-Balance Volume (OBV)
def compute_obv(df, column='Close', volume_column='Volume'):
    obv = (np.sign(df[column].diff()) * df[volume_column]).fillna(0).cumsum()
    df['OBV'] = obv
    return df

# Average True Range (ATR)
def compute_atr(df, high_col='High', low_col='Low', close_col='Close', window=14):
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift(1))
    low_close = np.abs(df[low_col] - df[close_col].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    df['ATR'] = atr
    return df
