import streamlit as st

st.set_page_config(page_title="Trading Signal Dashboard (Tabs)")

st.markdown("""
<style>
/* ===== Brand palette & tokens ===== */
:root {
  --pink: #dc3f6c;
  --teal: #719e99;
  --ink: #151516;
  --bg: #f7f8f9;
  --card: #fff;
  --line: #e6e8eb;
  --radius: 16px;
  --shadow: 0 6px 18px rgba(0,0,0,.06);
}

/* ===== Layout ===== */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--ink);
}
.block-container {
  max-width: 1200px;
  padding: 1.5rem 1rem 3rem;
}

/* ===== Headings / text ===== */
h1, h2, h3 { 
  color: var(--pink);
  letter-spacing: .2px;
}
hr { border-color: var(--line); }

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#fff 0%,#f3f6f6 100%);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
  color: var(--ink) !important;
}

/* ===== Metric cards ===== */
div[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1rem;
}
div[data-testid="stMetricValue"] {
  color: var(--pink);
  font-weight: 700;
  font-size: 1.7rem;
}

/* ===== Buttons ===== */
.stButton>button {
  background: var(--pink);
  color: #fff;
  border: 0;
  border-radius: 12px;
  padding: .6rem 1rem;
  box-shadow: var(--shadow);
  cursor: pointer;
}
.stButton>button:hover { filter: brightness(.95); }

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: .5rem 1rem;
  margin-right: .4rem;
  border: 1px solid transparent;
  background: transparent;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: var(--pink);
  border-color: var(--pink);
  color: #fff;
}
.stTabs [data-baseweb="tab"]:hover { border-color: var(--pink); }

/* ===== Expanders ===== */
[data-testid="stExpander"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
[data-testid="stExpander"] summary {
  color: var(--pink);
  font-weight: 600;
}

/* ===== Tables ===== */
[data-testid="stDataFrame"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
[data-testid="stDataFrame"] thead th {
  background: #fafbfc;
  font-weight: 600;
}

/* ===== Plots ===== */
[data-testid="stPlotlyChart"],
.js-plotly-plot {
  background: var(--card) !important;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: .4rem;
}

/* Uncomment to hide menu/footer
#MainMenu {display: none;}
footer {display: none;}
*/
</style>
""", unsafe_allow_html=True)





import streamlit as st, yfinance as yf, datetime, matplotlib.pyplot as plt
import plotly.graph_objects as go, pandas as pd, numpy as np
from typing import Dict

from indicators import add_moving_averages, add_rsi, add_macd, add_returns
from signals import add_trading_signals, strategy_returns, equity_curve, signal_to_position
from analytics import summarize_trades, performance_report, trade_stats
from sarima import suggest_sarima_orders, infer_regular_series
from statsmodels.tsa.statespace.sarimax import SARIMAX


st.sidebar.title("Signal Generator")

ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value="TSLA:100,NVDA:100")
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

# Parse input into {ticker: quantity}
portfolio = {}
for item in ticker_input.split(","):
    item = item.strip()
    if ":" in item:
        t, q = item.split(":")
        portfolio[t.strip().upper()] = int(q.strip())
    else:
        portfolio[item.strip().upper()] = 0  # default 0 if no quantity

raw_tickers = list(portfolio.keys())
if not raw_tickers:
    st.warning("Please provide at least one ticker.")
    st.stop()

# --- Simple Portfolio Value ---
st.sidebar.subheader("Portfolio Value")
total_value = 0
for ticker, qty in portfolio.items():
    try:
        price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        holding_value = qty * price
        total_value += holding_value
        st.sidebar.write(f"{ticker}: {qty} × ${price:.2f} = ${holding_value:,.2f}")
    except Exception:
        st.sidebar.write(f"{ticker}: data unavailable")

st.sidebar.metric("Total Value", f"${total_value:,.2f}")


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

        st.subheader("SARIMA Forecast")
        fc_steps = st.slider(f"Forecast horizon (periods) — {ticker}", min_value=5, max_value=120, value=30, step=5, key=f"fc_{ticker}")

        try:
            y_raw = df['Close'].dropna().astype(float)
            (order_pdq, seasonal_pdqs, freq) = suggest_sarima_orders(y_raw)
            p, d, q = order_pdq
            P, D, Q, s = seasonal_pdqs
            st.caption(f"Suggested SARIMA order: (p,d,q)=({p},{d},{q}), seasonal (P,D,Q,s)=({P},{D},{Q},{s})")

            y, _ = infer_regular_series(y_raw)
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
