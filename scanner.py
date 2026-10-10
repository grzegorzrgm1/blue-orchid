import json, math, time
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
TICKERS_FILE = ROOT / "tickers.csv"
OUT_FILE = ROOT / "data.json"

# V1: prosty, stały model. Bez EMA.
# Timeframe'y: W / D / 4H / 1H.
# 4H jest budowane z danych 1H.

def fetch(ticker):
    t = yf.Ticker(ticker)
    daily = t.history(period="2y", interval="1d", auto_adjust=True)
    hourly = t.history(period="60d", interval="1h", auto_adjust=True)
    if daily.empty or hourly.empty:
        return None
    four = hourly.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    weekly = daily.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return weekly, daily, four, hourly

def hh_hl(df, lookback=5):
    if len(df) < lookback + 3: return False
    recent = df.tail(lookback)
    prev = df.iloc[-lookback*2:-lookback]
    return recent["High"].max() >= prev["High"].max() and recent["Low"].min() >= prev["Low"].min()

def score_one(ticker, sector):
    got = fetch(ticker)
    if got is None: return None
    w,d,h4,h1 = got
    close = float(d["Close"].iloc[-1])
    sma50 = float(d["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(d["Close"].rolling(200).mean().iloc[-1])
    atr = float((d["High"]-d["Low"]).rolling(14).mean().iloc[-1])
    if any(math.isnan(x) for x in [sma50,sma200,atr]): return None

    wtrend = hh_hl(w) and float(w["Close"].iloc[-1]) > float(w["Close"].rolling(20).mean().iloc[-1])
    dtrend = hh_hl(d) and close > sma50 > sma200
    h4trend = hh_hl(h4)

    # Prostą miarą RS: ostatnie 60 sesji.
    rs = float(d["Close"].pct_change(60).iloc[-1])

    # Pullback: cena nie dalej niż 2 ATR od SMA50 i nie pod SMA200.
    pullback = abs(close - sma50) <= 2.0 * atr and close > sma200
    breakout = close >= float(d["High"].tail(20).iloc[:-1].max())

    points = 0
    points += 25 if wtrend else 0
    points += 25 if dtrend else 0
    points += 15 if h4trend else 0
    points += 15 if rs > 0.08 else (8 if rs > 0 else 0)
    points += 10 if pullback or breakout else 0
    points += 10 if float(h1["Volume"].tail(5).mean()) > float(h1["Volume"].tail(20).mean()) else 0

    if points < 60:
        return None

    setup = "Breakout" if breakout else ("Pullback" if pullback else "Trend")
    return {
        "ticker": ticker, "score": points, "sector": sector,
        "weekly": "HH/HL" if wtrend else "Mixed",
        "daily": "HH/HL" if dtrend else "Mixed",
        "setup4h": setup,
        "entry1h": "Confirm" if h4trend else "Watch",
        "rs": "A+" if rs > .15 else ("A" if rs > .08 else "B")
    }

def main():
    universe = pd.read_csv(TICKERS_FILE)
    results = []
    for _, row in universe.iterrows():
        try:
            x = score_one(row["ticker"], row["sector"])
            if x: results.append(x)
        except Exception as e:
            print("skip", row["ticker"], e)
    results.sort(key=lambda x: x["score"], reverse=True)
    OUT_FILE.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M UTC"), "setups": results[:30]}, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} setups to {OUT_FILE}")

if __name__ == "__main__":
    main()
