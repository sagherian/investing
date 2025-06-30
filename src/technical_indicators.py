### Functions for Data Prep ###
import pandas as pd
import numpy as np

def enrich_with_technical_indicators(df):
    df = compute_percent_change(df)                     # Percent change
    df = compute_sma(df)                                # Simple Moving Averages
    df = compute_ema(df)                                # Exponential Moving Averages
    df = compute_macd(df)                               # MACD
    df = compute_bollinger_bands(df)                    # Bollinger Bands
    df = compute_rsi(df)                                # RSI
    df = compute_obv(df)                                # On-Balance Volume
    df = compute_atr(df)                                # Average True Range
    df = compute_mfi(df)                                # Money Flow Index
    df = compute_historical_volatility(df)              # Historical Volatility
    df = compute_donchian_channels(df)                  # Donchian Channels
    df = compute_z_score(df)                            # Z-Score
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

import pandas as pd
import numpy as np

# Percent Change
def compute_percent_change(df, column='Close', horizons=[1, 5, 10]):
    for h in horizons:
        df[f'Pct_Change_{h}'] = df[column].pct_change(h) * 100
    return df

# Simple Moving Average (SMA)
def compute_sma(df, column='Close', windows=[10, 20, 50, 100, 200]):
    for window in windows:
        df[f'SMA_{window}'] = df[column].rolling(window).mean()
    return df

# Exponential Moving Average (EMA)
def compute_ema(df, column='Close', windows=[10, 20, 50, 100, 200]):
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

# Relative Strength Index (RSI)
def compute_rsi(df, column='Close', window=14):
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# On-Balance Volume (OBV)
def compute_obv(df, column='Close', volume_column='Volume'):
    obv = (np.sign(df[column].diff()) * df[volume_column]).fillna(0).cumsum()
    df['OBV'] = obv
    return df

# Average True Range (ATR)
def compute_atr(df, high_col='High', low_col='Low', close_col='Close', window=14):
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift())
    low_close = np.abs(df[low_col] - df[close_col].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window).mean()
    return df

# Money Flow Index (MFI)
def compute_mfi(df, high_col='High', low_col='Low', close_col='Close', volume_col='Volume', window=14):
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
    money_flow = typical_price * df[volume_col]
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    positive_sum = positive_flow.rolling(window).sum()
    negative_sum = negative_flow.rolling(window).sum()
    money_ratio = positive_sum / (negative_sum + 1e-9)
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    return df

# Historical Volatility
def compute_historical_volatility(df, column='Close', window=14):
    log_returns = np.log(df[column] / df[column].shift())
    df['Hist_Volatility'] = log_returns.rolling(window).std() * np.sqrt(window)
    return df

# Donchian Channels
def compute_donchian_channels(df, column='Close', window=20):
    df['Donchian_Upper'] = df[column].rolling(window).max()
    df['Donchian_Lower'] = df[column].rolling(window).min()
    return df

# Z-Score
def compute_z_score(df, column='Close', window=20):
    rolling_mean = df[column].rolling(window).mean()
    rolling_std = df[column].rolling(window).std()
    df['Z_Score'] = (df[column] - rolling_mean) / rolling_std
    return df
