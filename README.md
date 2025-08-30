# financial dashboard
A tool to analyze and visualize financial data (stocks, ETFs, crypto) using Yahoo Finance. It calculates technical indicators and generates buy/sell/hold signals

## Features
- **Multi-asset support**: Stocks, ETFs, and crypto (via Yahoo Finance)
- **Technical indicators**:
  - Moving Averages (SMA/EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
- **Risk profiles**:
  - Conservative: safer thresholds, earlier sell signals
  - Aggressive: higher tolerance, later sell signals
- **Portfolio tracking**:
  - Register current holdings (AAPL)
  - Personalized Buy/Hold/Sell recommendations
- **Visualizations**:
  - Price charts with indicator overlays
  - RSI, MACD, and other signals
  - SARIMA forecasting option
- **Company info**: basic stock metadata (industry, market cap, etc.)

## Tech Stack
- Python
- Streamlit
- yfinance
- pandas, numpy, matplotlib

## Installation
Clone the repository:

```bash
git clone https://github.com/hanbee-Y/financial_dashboard.git
cd financial_dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run main.py
```






