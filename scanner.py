import json
import math
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
TICKERS_FILE = ROOT / "tickers.csv"
OUT_FILE = ROOT / "data.json"

DIAGNOSTIC_TICKERS = {"VST", "BWXT", "OKLO", "CEG"}


def fetch(ticker):
    t = yf.Ticker(ticker)
    daily = t.history(period="2y", interval="1d", auto_adjust=True)
    hourly = t.history(period="60d", interval="1h", auto_adjust=True)

    if daily.empty or hourly.empty:
        return None

    four = hourly.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    weekly = daily.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    return weekly, daily, four, hourly


def hh_hl(df, lookback=5):
    if len(df) < lookback + 3:
        return False

    recent = df.tail(lookback)
    previous = df.iloc[-lookback * 2:-lookback]

    return (
        recent["High"].max() >= previous["High"].max()
        and recent["Low"].min() >= previous["Low"].min()
    )


def score_one(ticker, sector):
    got = fetch(ticker)

    if got is None:
        return {
            "ticker": ticker,
            "sector": sector,
            "data_ok": False,
            "score": None,
            "qualifies": False,
            "reason": "Brak danych dziennych lub godzinowych",
        }

    w, d, h4, h1 = got

    close = float(d["Close"].iloc[-1])
    sma50 = float(d["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(d["Close"].rolling(200).mean().iloc[-1])
    atr = float((d["High"] - d["Low"]).rolling(14).mean().iloc[-1])

    if any(math.isnan(x) for x in [sma50, sma200, atr]):
        return {
            "ticker": ticker,
            "sector": sector,
            "data_ok": False,
            "score": None,
            "qualifies": False,
            "reason": "Za mało danych do obliczenia SMA50/SMA200/ATR",
        }

    weekly_hhhl = hh_hl(w)
    weekly_sma20 = (
        len(w) >= 20
        and float(w["Close"].iloc[-1])
        > float(w["Close"].rolling(20).mean().iloc[-1])
    )

    daily_hhhl = hh_hl(d)
    h4_hhhl = hh_hl(h4)

    wtrend = weekly_hhhl and weekly_sma20
    dtrend = daily_hhhl and close > sma50 > sma200

    # Dotychczasowa miara: bezwzględny zwrot z 60 sesji.
    rs = float(d["Close"].pct_change(60).iloc[-1])

    pullback = abs(close - sma50) <= 2.0 * atr and close > sma200
    previous_high = float(d["High"].tail(20).iloc[:-1].max())
    breakout = close >= previous_high

    h1_volume_up = (
        len(h1) >= 20
        and float(h1["Volume"].tail(5).mean())
        > float(h1["Volume"].tail(20).mean())
    )

    # Zachowana dotychczasowa punktacja.
    points = 0
    points += 25 if wtrend else 0
    points += 25 if dtrend else 0
    points += 15 if h4_hhhl else 0
    points += 15 if rs > 0.08 else (8 if rs > 0 else 0)
    points += 10 if pullback or breakout else 0
    points += 10 if h1_volume_up else 0

    setup = "Breakout" if breakout else (
        "Pullback" if pullback else "Trend"
    )
    qualifies = points >= 60

    failed = []
    if not wtrend:
        failed.append("trend W1")
    if not dtrend:
        failed.append("trend D1 / cena > SMA50 > SMA200")
    if not h4_hhhl:
        failed.append("układ HH/HL H4")
    if rs <= 0:
        failed.append("dodatni zwrot z 60 sesji")
    if not (pullback or breakout):
        failed.append("Pullback lub Breakout")
    if not h1_volume_up:
        failed.append("wzrost wolumenu H1")
    if not qualifies:
        failed.append(f"próg 60 punktów; wynik {points}")

    return {
        "ticker": ticker,
        "sector": sector,
        "data_ok": True,
        "score": points,
        "qualifies": qualifies,
        "weekly": "HH/HL" if wtrend else "Mixed",
        "daily": "HH/HL" if dtrend else "Mixed",
        "h4_hhhl": h4_hhhl,
        "setup4h": setup,
        "entry1h": "Confirm" if h4_hhhl else "Watch",
        "h1_volume_up": h1_volume_up,
        "close": round(close, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "atr14": round(atr, 2),
        "above_sma50": close > sma50,
        "above_sma200": close > sma200,
        "sma50_above_sma200": sma50 > sma200,
        "return_60d_pct": round(rs * 100, 2),
        "rs": "A+" if rs > 0.15 else ("A" if rs > 0.08 else "B"),
        "pullback": pullback,
        "breakout": breakout,
        "failed_conditions": failed,
    }


def main():
    universe = pd.read_csv(TICKERS_FILE)
    results = []
    errors = []

    for _, row in universe.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        sector = str(row["sector"]).strip()

        try:
            item = score_one(ticker, sector)
            results.append(item)

            if not item.get("data_ok", False):
                errors.append({
                    "ticker": ticker,
                    "reason": item.get("reason", "Błąd danych"),
                })

        except Exception as e:
            message = str(e)
            print("skip", ticker, message)
            errors.append({"ticker": ticker, "reason": message})
            results.append({
                "ticker": ticker,
                "sector": sector,
                "data_ok": False,
                "score": None,
                "qualifies": False,
                "reason": message,
            })

    setups = [
        x for x in results
        if x.get("data_ok") and x.get("qualifies")
    ]
    setups.sort(key=lambda x: x["score"], reverse=True)

    diagnostics = [
        x for x in results
        if x["ticker"] in DIAGNOSTIC_TICKERS
    ]
    diagnostics.sort(key=lambda x: x["ticker"])

    results.sort(
        key=lambda x: (
            x.get("score") is not None,
            x.get("score") or -1,
        ),
        reverse=True,
    )

    output = {
        "updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "universe_count": len(universe),
        "scanned_count": sum(
            1 for x in results if x.get("data_ok")
        ),
        "qualifying_count": len(setups),
        "setups": setups[:30],
        "diagnostic_tickers": diagnostics,
        "all_diagnostics": results,
        "data_errors": errors,
    }

    OUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Scanned {output['scanned_count']}/{len(universe)} tickers; "
        f"saved {len(setups)} qualifying setups to {OUT_FILE}"
    )
    print(
        "Diagnostic tickers:",
        json.dumps(diagnostics, ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
