import pandas as pd
import numpy as np

def add_moving_averages(df, short=20, long=50):
    df['SMA'] = df['Close'].rolling(window=short).mean()
    df['EMA'] = df['Close'].ewm(span=long, adjust=False).mean()
    return df


def add_rsi(df, window= 14):
    from ta.momentum import RSIIndicator
    rsi = RSIIndicator(close=df['Close'], window=window)
    df['RSI'] = rsi.rsi()
    return df


def add_macd(df):
    from ta.trend import MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Hist_Prev'] = df['MACD_Hist'].shift(1)
    return df


def add_returns(df):
    df['pct_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close']).diff()
    return df
