import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

def infer_regular_series(y: pd.Series) -> tuple[pd.Series, str]:
    y = y.dropna().astype(float)
    freq = pd.infer_freq(y.index)
    if freq is None:
        freq = 'B' 
        y = y.asfreq(freq).ffill()
    else:
        y = y.asfreq(freq)
    return y, freq


def suggest_d(y: pd.Series, max_d: int = 2) -> int:
    d, tmp = 0, y.copy()
    while d < max_d:
        pval = adfuller(tmp, autolag='AIC')[1]
        if pval < 0.05:
            break
        tmp = tmp.diff().dropna()
        d += 1
    return d


def choose_seasonal_period(y: pd.Series, candidates=(5, 7, 12, 21, 30, 52)) -> int:
    n = len(y)
    if n < 60:
        return 5
    acf_vals = sm_acf(y - y.mean(), nlags=max(candidates), fft=True)
    best_s = max(candidates, key=lambda s: abs(acf_vals[s]) if s < len(acf_vals) else 0.0)
    return best_s


def suggest_D(y: pd.Series, s: int) -> int:
    ys = (y - y.shift(s)).dropna()
    if len(ys) < 30:
        return 0
    pval = adfuller(ys, autolag='AIC')[1]
    return 1 if pval >= 0.05 else 0


def suggest_pq(tmp: pd.Series, max_lag: int = 20) -> tuple[int, int]:
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


def suggest_PQ(tmp: pd.Series, s: int) -> tuple[int, int]:
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
    y, freq = infer_regular_series(y_raw)
    d = suggest_d(y, max_d=2)
    tmp = y.diff(d).dropna() if d > 0 else y.copy()
    s = choose_seasonal_period(tmp)
    D = suggest_D(tmp, s)
    tmp2 = (tmp - tmp.shift(s)).dropna() if D == 1 else tmp
    p, q = suggest_pq(tmp2)
    P, Q = suggest_PQ(tmp2, s)
    return (p, d, q), (P, D, Q, s), freq