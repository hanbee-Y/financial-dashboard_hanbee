# financial dashboard
A tool to analyze and visualize financial data (stocks, ETFs, crypto) using Yahoo Finance. It calculates technical indicators and generates buy/sell/hold signals

## Features
- **Multi-asset support**: Stocks, ETFs, and crypto (via Yahoo Finance)
- **Technical indicators**:
  - Moving Averages (SMA/EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- **Risk profiles**:
  - Conservative: safer thresholds
  - Aggressive: higher tolerance
- **Portfolio tracking**:
  - Register current holdings (example - TSLA:100, NVDA:100)
  - Personalized Buy/Hold/Sell recommendations
- **Visualizations**:
  - Price charts
  - SARIMA forecasting option
- **Company info**: basic stock metadata (industry, market cap, etc.)

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
streamlit run app.py
```






