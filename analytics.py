import pandas as pd
import numpy as np
from signals import signal_to_position

def summarize_trades(df, signal_col: str):
    pos = signal_to_position(df[signal_col])
    pos_shift = pos.shift(1).fillna(0)
    entries = (pos == 1) & (pos_shift == 0)
    exits = (pos == 0) & (pos_shift == 1)

    trades = []
    for dt, is_entry in entries.items():
        if is_entry:
            subsequent_exits = exits[exits.index > dt]
            if subsequent_exits.any():
                exit_dt = subsequent_exits.index[0]
            else:
                exit_dt = df.index[-1]
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
            'Cumulative Return (%)','Annual Return (%)','Annual Volatility (%)','Sharpe',
            'Sortino','Max Drawdown (%)'
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


def trade_stats(trades) :
    if trades.empty:
        return {
            'Trades': 0,
            'Winners': 0,
            'Losers': 0,
            'Win Rate (%)': np.nan,
            'Profit Factor': np.nan,
            'Avg Holding (days)': np.nan,
        }
    wins = (trades['Return'] > 0).sum()
    losses = (trades['Return'] <= 0).sum()
    win_rate = wins / len(trades) * 100 if len(trades) > 0 else np.nan
    gross_profit = trades.loc[trades['Return'] > 0, 'Return'].sum()
    gross_loss = -trades.loc[trades['Return'] <= 0, 'Return'].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    avg_hold = trades['HoldingDays'].mean() if not trades['HoldingDays'].empty else np.nan
    return {
        'Trades': int(len(trades)),
        'Winners': int(wins),
        'Losers': int(losses),
        'Win Rate (%)': round(win_rate, 2) if not np.isnan(win_rate) else np.nan,
        'Profit Factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
        'Avg Holding (days)': round(avg_hold, 1) if not np.isnan(avg_hold) else np.nan,
    }