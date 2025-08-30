import pandas as pd
import numpy as np

def add_trading_signals(df, rsi_buy=30, rsi_sell=70):
    # RSI
    df['RSI_Signal'] = 'Hold'
    df.loc[df['RSI'] < rsi_buy, 'RSI_Signal'] = 'Buy'
    df.loc[df['RSI'] > rsi_sell, 'RSI_Signal'] = 'Sell'

    # MACD
    df['MACD_SignalFlag'] = 'Hold'
    df.loc[(df['MACD_Hist'] > 0) & (df['MACD_Hist_Prev'] <= 0), 'MACD_SignalFlag'] = 'Buy'
    df.loc[(df['MACD_Hist'] < 0) & (df['MACD_Hist_Prev'] >= 0), 'MACD_SignalFlag'] = 'Sell'

    # MA
    df['SMA_above_EMA'] = df['SMA'] > df['EMA']
    df['MA_Up_Trend'] = df['SMA_above_EMA'] & df['SMA_above_EMA'].shift(1)
    df['SMA_below_EMA'] = df['SMA'] < df['EMA']
    df['MA_Down_Trend'] = df['SMA_below_EMA'] & df['SMA_below_EMA'].shift(1)
    df['MA_Signal'] = 'Hold'
    df.loc[(df['MA_Up_Trend']) & (~df['MA_Up_Trend'].shift(1).fillna(False)), 'MA_Signal'] = 'Buy'
    df.loc[(df['MA_Down_Trend']) & (~df['MA_Down_Trend'].shift(1).fillna(False)), 'MA_Signal'] = 'Sell'
    return df


def signal_to_position(signal_series: pd.Series):
    pos = pd.Series(0, index=signal_series.index, dtype=float)
    in_pos = False
    for idx, sig in signal_series.fillna('Hold').items():
        if sig == 'Buy':
            in_pos = True
        elif sig == 'Sell':
            in_pos = False
        pos.loc[idx] = 1.0 if in_pos else 0.0
    return pos

def strategy_returns(df, signal_col, ret_col='log_return'):
    pos = signal_to_position(df[signal_col])
    strat_ret = pos.shift(1).fillna(0) * df[ret_col]
    return strat_ret.rename(f"{signal_col}_returns")

def equity_curve(returns, kind='log'):
    if kind == 'log':
        curve = returns.cumsum().apply(np.exp)
    else:
        curve = (1 + returns).cumprod()
    return curve.rename(returns.name.replace('_returns', '_equity'))

def summarize_trades(df, signal_col):
    pos = signal_to_position(df[signal_col])
    pos_shift = pos.shift(1).fillna(0)
    entries = (pos == 1) & (pos_shift == 0)
    exits = (pos == 0) & (pos_shift == 1)

    trades = []
    for dt, is_entry in entries.items():
        if is_entry:
            subsequent_exits = exits[exits.index > dt]
            exit_dt = subsequent_exits.index[0] if subsequent_exits.any() else df.index[-1]
            returns_slice = df.loc[dt:exit_dt, 'log_return']
            trade_log_ret = returns_slice.iloc[1:].sum()
            trade_ret = np.exp(trade_log_ret) - 1
            holding_days = (exit_dt - dt).days
            trades.append({
                'Entry': dt,
                'Exit': exit_dt,
                'Return': trade_ret,
                'HoldingDays': holding_days,
            })
    return pd.DataFrame(trades)

def performance_report(returns, rf=0.0):
    if returns.dropna().empty:
        return {k: np.nan for k in [
            'Cumulative Return (%)','Annual Return (%)','Annual Volatility (%)',
            'Sharpe','Sortino','Max Drawdown (%)'
        ]}

    daily_log = returns.fillna(0)
    daily_lin = np.exp(daily_log) - 1
    cum_ret = np.exp(daily_log.cumsum()).iloc[-1] - 1

    ann_factor = 252
    ann_ret = (1 + daily_lin.mean()) ** ann_factor - 1
    ann_vol = daily_lin.std() * np.sqrt(ann_factor)

    daily_excess = daily_lin - (rf / ann_factor)
    sharpe = (daily_excess.mean() / daily_lin.std()) * np.sqrt(ann_factor) if daily_lin.std() != 0 else np.nan

    downside = daily_lin[daily_lin < 0]
    downside_std = downside.std()
    sortino = (daily_lin.mean() * np.sqrt(ann_factor)) / (downside_std * np.sqrt(ann_factor)) if downside_std not in [0, np.nan] and not np.isnan(downside_std) else np.nan

    eq = (1 + daily_lin).cumprod()
    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    mdd = dd.min()

    return {
        'Cumulative Return (%)': round(cum_ret * 100, 2),
        'Annual Return (%)': round(ann_ret * 100, 2),
        'Annual Volatility (%)': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 2) if not np.isnan(sharpe) else np.nan,
        'Sortino': round(sortino, 2) if not np.isnan(sortino) else np.nan,
        'Max Drawdown (%)': round(mdd * 100, 2),
    }

def trade_stats(trades):
    if trades.empty:
        return {'Trades': 0,'Winners': 0,'Losers': 0,
                'Win Rate (%)': np.nan,'Profit Factor': np.nan,'Avg Holding (days)': np.nan}
    wins = (trades['Return'] > 0).sum()
    losses = (trades['Return'] <= 0).sum()
    win_rate = wins / len(trades) * 100
    gross_profit = trades.loc[trades['Return'] > 0, 'Return'].sum()
    gross_loss = -trades.loc[trades['Return'] <= 0, 'Return'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    avg_hold = trades['HoldingDays'].mean()
    return {
        'Trades': int(len(trades)),
        'Winners': int(wins),
        'Losers': int(losses),
        'Win Rate (%)': round(win_rate, 2),
        'Profit Factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
        'Avg Holding (days)': round(avg_hold, 1),
    }
