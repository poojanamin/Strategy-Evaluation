"""
Implementing Technical Indicators:
    1. Relative Strength Index (RSI)
    2. Moving Average Convergence Divergence (MACD)
    3. Bollinger Bands

@Name : Pooja Amin
@UserID : pamin33

"""

import pandas as pd
import numpy as np


def moving_average(data, window=20):
    """ Calculate the Moving Average for the given data. """
    return data.rolling(window=window).mean()


def rsi(data, window=14):
    """ Compute the Relative Strength Index (RSI). """
    diff = data.diff(1)
    gain = (diff.where(diff > 0, 0)).fillna(0)
    loss = (-diff.where(diff < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(data, slow=26, fast=12):
    """ Compute the Moving Average Convergence Divergence (MACD). """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def bollinger_bands(data, window=20):
    """ Calculate Bollinger Bands. """
    ma = moving_average(data, window)
    std = data.rolling(window=window).std()
    upper_band = ma + (std * 2)
    lower_band = ma - (std * 2)
    return upper_band, lower_band


def author():
    return 'pamin33'


# Example usage:
if __name__ == "__main__":
    # Generate example data
    dates = pd.date_range('2020-01-01', '2020-12-31')
    np.random.seed(42)
    data = pd.Series(np.random.normal(100, 10, size=len(dates)), index=dates)

    # Calculate indicators
    data_ma = moving_average(data)
    data_rsi = rsi(data)
    macd_line, signal_line = macd(data)
    upper_band, lower_band = bollinger_bands(data)

    # Print example outputs
    print("Moving Average:\n", data_ma.tail())
    print("RSI:\n", data_rsi.tail())
    print("MACD Line:\n", macd_line.tail())
    print("Signal Line:\n", signal_line.tail())
    print("Bollinger Bands:\n", upper_band.tail(), lower_band.tail())
