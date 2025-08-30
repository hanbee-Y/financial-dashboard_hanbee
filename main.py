import streamlit as st
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict


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

#------------------------------------------------------------

def add_trading_signals(df, rsi_buy= 30, rsi_sell = 70):
    df['RSI_Signal'] = 'Hold'
    df.loc[df['RSI'] < rsi_buy, 'RSI_Signal'] = 'Buy'
    df.loc[df['RSI'] > rsi_sell, 'RSI_Signal'] = 'Sell'

    df['MACD_SignalFlag'] = 'Hold'
    df.loc[(df['MACD_Hist'] > 0) & (df['MACD_Hist_Prev'] <= 0), 'MACD_SignalFlag'] = 'Buy'
    df.loc[(df['MACD_Hist'] < 0) & (df['MACD_Hist_Prev'] >= 0), 'MACD_SignalFlag'] = 'Sell'

    df['SMA_above_EMA'] = df['SMA'] > df['EMA']
    df['MA_Up_Trend'] = df['SMA_above_EMA'] & df['SMA_above_EMA'].shift(1)
    df['SMA_below_EMA'] = df['SMA'] < df['EMA']
    df['MA_Down_Trend'] = df['SMA_below_EMA'] & df['SMA_below_EMA'].shift(1)
    df['MA_Signal'] = 'Hold'
    df.loc[(df['MA_Up_Trend']) & (~df['MA_Up_Trend'].shift(1).fillna(False)), 'MA_Signal'] = 'Buy'
    df.loc[(df['MA_Down_Trend']) & (~df['MA_Down_Trend'].shift(1).fillna(False)), 'MA_Signal'] = 'Sell'

    return df


def signal_to_position(signal_series):
    pos = pd.Series(0, index=signal_series.index, dtype=float)
    in_pos = False
    for idx, sig in signal_series.fillna('Hold').items():
        if sig == 'Buy':
            in_pos = True
        elif sig == 'Sell':
            in_pos = False
        pos.loc[idx] = 1.0 if in_pos else 0.0
    return pos


def strategy_returns(df, signal_col: str, ret_col: str = 'log_return'):
    pos = signal_to_position(df[signal_col])
    strat_ret = pos.shift(1).fillna(0) * df[ret_col]
    return strat_ret.rename(f"{signal_col}_returns")


def equity_curve(returns, kind: str = 'log'):
    if kind == 'log':
        curve = returns.cumsum().apply(np.exp)
    else:
        curve = (1 + returns).cumprod()
    return curve.rename(returns.name.replace('_returns', '_equity'))


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

#SARIMA
from statsmodels.tsa.stattools import adfuller, acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _infer_regular_series(y: pd.Series) -> tuple[pd.Series, str]:
    y = y.dropna().astype(float)
    freq = pd.infer_freq(y.index)
    if freq is None:
        freq = 'B' 
        y = y.asfreq(freq).ffill()
    else:
        y = y.asfreq(freq)
    return y, freq


def _suggest_d(y: pd.Series, max_d: int = 2) -> int:
    d, tmp = 0, y.copy()
    while d < max_d:
        pval = adfuller(tmp, autolag='AIC')[1]
        if pval < 0.05:
            break
        tmp = tmp.diff().dropna()
        d += 1
    return d


def _choose_seasonal_period(y: pd.Series, candidates=(5, 7, 12, 21, 30, 52)) -> int:
    n = len(y)
    if n < 60:
        return 5
    acf_vals = sm_acf(y - y.mean(), nlags=max(candidates), fft=True)
    best_s = max(candidates, key=lambda s: abs(acf_vals[s]) if s < len(acf_vals) else 0.0)
    return best_s


def _suggest_D(y: pd.Series, s: int) -> int:
    ys = (y - y.shift(s)).dropna()
    if len(ys) < 30:
        return 0
    pval = adfuller(ys, autolag='AIC')[1]
    return 1 if pval >= 0.05 else 0


def _suggest_pq(tmp: pd.Series, max_lag: int = 20) -> tuple[int, int]:
    n = len(tmp)
    if n < 10:
        return (1, 1)
    lag = min(max_lag, max(5, n // 4))
    thr = 1.96 / np.sqrt(n)
    acf_vals = sm_acf(tmp, nlags=lag, fft=True)[1:]
    pacf_vals = sm_pacf(tmp, nlags=lag, method='ywmle')[1:]

    def count_sig(seq):
        c = 0
        for v in seq:
            if abs(v) > thr:
                c += 1
            else:
                break
        return c

    p = max(1, min(3, count_sig(pacf_vals)))
    q = max(1, min(3, count_sig(acf_vals)))
    return (p, q)


def _suggest_PQ(tmp: pd.Series, s: int) -> tuple[int, int]:
    n = len(tmp)
    if n < s * 3:
        return (0, 0)
    thr = 1.96 / np.sqrt(n)
    acf_vals = sm_acf(tmp, nlags=s, fft=True)
    pacf_vals = sm_pacf(tmp, nlags=s, method='ywmle')
    P = 1 if (len(pacf_vals) > s and abs(pacf_vals[s]) > thr) else 0
    Q = 1 if (len(acf_vals) > s and abs(acf_vals[s]) > thr) else 0
    return (P, Q)


def suggest_sarima_orders(y_raw: pd.Series) -> tuple[tuple[int,int,int], tuple[int,int,int,int], str]:
    y, freq = _infer_regular_series(y_raw)
    d = _suggest_d(y, max_d=2)
    tmp = y.diff(d).dropna() if d > 0 else y.copy()
    s = _choose_seasonal_period(tmp)
    D = _suggest_D(tmp, s)
    tmp2 = (tmp - tmp.shift(s)).dropna() if D == 1 else tmp
    p, q = _suggest_pq(tmp2)
    P, Q = _suggest_PQ(tmp2, s)
    return (p, d, q), (P, D, Q, s), freq



st.set_page_config(page_title="Trading Signal Dashboard (Tabs)", layout="wide")

st.sidebar.title("Signal Generator")

ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value="TSLA,NVDA")
start_date = st.sidebar.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

short_window = st.sidebar.slider("Short MA window (SMA)", min_value=5, max_value=100, value=20)
long_window = st.sidebar.slider("Long MA window (EMA)", min_value=10, max_value=200, value=50)

risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Aggressive"], index=0)

indicator_choices = st.sidebar.multiselect(
    "Indicators for Consensus",
    ["RSI", "MACD", "MA"],
    default=["RSI", "MACD", "MA"]
)

if risk_profile == "Conservative":
    rsi_buy, rsi_sell = 35, 70
else:
    rsi_buy, rsi_sell = 25, 80

st.title("Financial Dashboard")
st.markdown("_Data processing in python project made by Hanbee Yoo_")
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
if not raw_tickers:
    st.warning("Please provide at least one ticker.")
    st.stop()

portfolio_equity_fig = go.Figure()

tabs = st.tabs(raw_tickers)

for i, ticker in enumerate(raw_tickers):
    with tabs[i]:
        st.header(f"{ticker}")
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info  
            company_name = info.get("longName", ticker)
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            market_cap = info.get("marketCap", None)

            st.subheader("Company Profile")
            st.write(f"**{company_name}**")
            st.write(f"**Sector:** {sector}")
            st.write(f"**Industry:** {industry}")
            if market_cap:
                st.write(f"**Market Cap:** {market_cap:,}")
        except Exception as e:
            st.warning(f"Could not fetch company info for {ticker}: {e}")

        df = yf.download(ticker, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        if df.empty:
            st.warning(f"No data found for {ticker}.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=0, axis=1)
            except Exception:
                try:
                    df = df.xs(ticker, level='Ticker', axis=1)
                except Exception:
                    df.columns = [c[0] for c in df.columns]
        
        df = add_moving_averages(df, short=short_window, long=long_window)
        df = add_rsi(df)
        df = add_macd(df)
        df = add_returns(df)
        df = add_trading_signals(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)

        signal_cols = {
            'RSI': 'RSI_Signal',
            'MACD': 'MACD_SignalFlag',
            'MA': 'MA_Signal',
        }

        for key, sig_col in signal_cols.items():
            strat_ret = strategy_returns(df, sig_col, 'log_return')
            df[strat_ret.name] = strat_ret
            df[equity_curve(strat_ret).name] = equity_curve(strat_ret)

        def row_consensus(row):
            if not indicator_choices:
                return 'Hold'
            votes = [row[signal_cols[ind]] for ind in indicator_choices]
            if votes.count('Buy') > votes.count('Sell') and votes.count('Buy') >= 1:
                return 'Buy'
            if votes.count('Sell') > votes.count('Buy') and votes.count('Sell') >= 1:
                return 'Sell'
            return 'Hold'

        df['Consensus_Signal'] = df.apply(row_consensus, axis=1)
        cons_ret = strategy_returns(df, 'Consensus_Signal', 'log_return')
        df['Consensus_returns'] = cons_ret
        df['Consensus_equity'] = equity_curve(cons_ret)

        consensus_now = df['Consensus_Signal'].iloc[-1]
        rec_map = {'Buy': 'BUY', 'Sell': 'SELL', 'Hold': 'NEUTRAL'}
        st.subheader("Current Recommendation (Consensus)")
        st.info(f"{rec_map.get(consensus_now, 'NEUTRAL')} — based on {', '.join(indicator_choices)}")

        with st.expander("Latest Indicator Signals"):
            st.write(pd.DataFrame({
                'Indicator': list(signal_cols.keys()),
                'Latest Signal': [df[col].iloc[-1] for col in signal_cols.values()],
            }))

        strat_metrics: Dict[str, Dict[str, float]] = {}
        for key, sig_col in signal_cols.items():
            m = performance_report(df[f'{sig_col}_returns'])
            strat_metrics[key] = m
        strat_metrics['Consensus'] = performance_report(df['Consensus_returns'])
        st.subheader("Risk & Performance Metrics")
        st.dataframe(pd.DataFrame(strat_metrics).T)

        st.subheader("Trade Aggregation (Consensus)")
        trades_df = summarize_trades(df, 'Consensus_Signal')
        st.dataframe(trades_df)
        ts = trade_stats(trades_df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", ts['Trades'])
        c2.metric("Win Rate", f"{ts['Win Rate (%)']}%" if not np.isnan(ts['Win Rate (%)']) else "—")
        c3.metric("Profit Factor", ts['Profit Factor'] if not np.isnan(ts['Profit Factor']) else "—")
        c4.metric("Avg Hold (days)", ts['Avg Holding (days)'] if not np.isnan(ts['Avg Holding (days)']) else "—")
        c5.metric("Cum. Return (Consensus)", f"{strat_metrics['Consensus']['Cumulative Return (%)']}%")

        st.subheader("Price with Buy/Sell (Consensus)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df['Close'], label='Close', linewidth=1.5)
        ax.plot(df.index, df['SMA'], label='SMA', linestyle='--', alpha=0.7)
        ax.plot(df.index, df['EMA'], label='EMA', linestyle='--', alpha=0.7)
        min_gap = 5
        buy = df[df['Consensus_Signal'] == 'Buy']
        sell = df[df['Consensus_Signal'] == 'Sell']
        buy = buy[buy.index.to_series().diff().dt.days.fillna(min_gap) >= min_gap]
        sell = sell[sell.index.to_series().diff().dt.days.fillna(min_gap) >= min_gap]
        ax.scatter(buy.index, buy['Close'], marker='^', s=50, label='Buy')
        ax.scatter(sell.index, sell['Close'], marker='v', s=50, label='Sell')
        ax.set_title(f"{ticker} — Price & Consensus Signals")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Strategy Equity Curves (log return based)")
        eq_fig = go.Figure()
        for key, sig_col in signal_cols.items():
            curve_col = f"{sig_col}_equity"
            eq_fig.add_trace(go.Scatter(x=df.index, y=df[curve_col], mode='lines', name=f"{key}"))
        eq_fig.add_trace(go.Scatter(x=df.index, y=df['Consensus_equity'], mode='lines', name='Consensus', line=dict(width=3)))
        eq_fig.update_layout(height=400, yaxis_title='Equity (normalized, start=1)', xaxis_title='Date')
        st.plotly_chart(eq_fig, use_container_width=True)

        portfolio_equity_fig.add_trace(go.Scatter(x=df.index, y=df['Consensus_equity'], mode='lines', name=f"{ticker} (Consensus)"))

        # --- SARIMA Forecast (per ticker tab) ---
        st.subheader("SARIMA Forecast")
        fc_steps = st.slider(f"Forecast horizon (periods) — {ticker}", min_value=5, max_value=120, value=30, step=5, key=f"fc_{ticker}")

        try:
            y_raw = df['Close'].dropna().astype(float)
            (order_pdq, seasonal_pdqs, freq) = suggest_sarima_orders(y_raw)
            p, d, q = order_pdq
            P, D, Q, s = seasonal_pdqs
            st.caption(f"Suggested SARIMA order: (p,d,q)=({p},{d},{q}), seasonal (P,D,Q,s)=({P},{D},{Q},{s})")

            y, _ = _infer_regular_series(y_raw)
            model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)

            fc = res.get_forecast(steps=fc_steps)
            fc_mean = fc.predicted_mean
            fc_ci = fc.conf_int(alpha=0.05)

            start_dt = y.index[-1] + pd.tseries.frequencies.to_offset(freq)
            future_index = pd.date_range(start=start_dt, periods=fc_steps, freq=freq)
            fc_mean.index = future_index
            fc_ci.index = future_index

            fig_fc, ax_fc = plt.subplots(figsize=(12, 5))
            ax_fc.plot(y.index, y.values, label="History", linewidth=1.6)
            ax_fc.plot(fc_mean.index, fc_mean.values, label="Forecast", linewidth=1.6)
            ax_fc.fill_between(fc_mean.index, fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values, alpha=0.25, label="95% CI")
            ax_fc.set_title(f"{ticker} — SARIMA Forecast (p,d,q)=({p},{d},{q}) x (P,D,Q,s)=({P},{D},{Q},{s})")
            ax_fc.set_xlabel("Date"); ax_fc.set_ylabel("Price")
            ax_fc.grid(True); ax_fc.legend()
            st.pyplot(fig_fc)

            fc_df = pd.DataFrame({
                "Forecast": fc_mean,
                "Lower (95%)": fc_ci.iloc[:, 0],
                "Upper (95%)": fc_ci.iloc[:, 1],
            }).round(2)
            st.dataframe(fc_df)
        except Exception as e:
            st.warning(f"SARIMA fitting failed for {ticker}: {e}")

if len(raw_tickers) > 1 and len(portfolio_equity_fig.data) > 0:
    st.header("Multi-Asset Consensus Equity Curves")
    portfolio_equity_fig.update_layout(height=420, yaxis_title='Equity (normalized, start=1)', xaxis_title='Date')
    st.plotly_chart(portfolio_equity_fig, use_container_width=True)
