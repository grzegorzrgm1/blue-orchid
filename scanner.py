import json
import math
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(".")
TICKERS_FILE = ROOT / "tickers.csv"
OUT_FILE = ROOT / "data.json"
DIAGNOSTIC_TICKERS = {"VST", "BWXT", "OKLO", "CEG"}
MIN_SCORE = 60


def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def fetch(ticker):
    stock = yf.Ticker(ticker)
    daily = stock.history(period="2y", interval="1d", auto_adjust=True)
    hourly = stock.history(period="60d", interval="1h", auto_adjust=True)
    if daily.empty or hourly.empty:
        return None

    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()

    weekly = daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    four_hour = hourly.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    return weekly, daily, four_hour, hourly


def confirmed_pivots(data, left=2, right=2):
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    high_pivots, low_pivots = [], []
    for i in range(left, len(data) - right):
        hw = highs[i-left:i+right+1]
        lw = lows[i-left:i+right+1]
        if highs[i] == hw.max() and (hw == highs[i]).sum() == 1:
            high_pivots.append((i, highs[i]))
        if lows[i] == lw.min() and (lw == lows[i]).sum() == 1:
            low_pivots.append((i, lows[i]))
    return high_pivots, low_pivots


def hh_hl(data, left=2, right=2, pivots_needed=2):
    """Dwa ostatnie potwierdzone szczyty i dołki muszą rosnąć."""
    highs, lows = confirmed_pivots(data, left, right)
    if len(highs) < pivots_needed or len(lows) < pivots_needed:
        return False
    last_highs = [v for _, v in highs[-pivots_needed:]]
    last_lows = [v for _, v in lows[-pivots_needed:]]
    return bool(
        all(b > a for a, b in zip(last_highs, last_highs[1:]))
        and all(b > a for a, b in zip(last_lows, last_lows[1:]))
    )


def failed_result(ticker, sector, reason):
    return {
        "ticker": ticker, "sector": sector, "data_ok": False,
        "score": None, "qualifies": False, "reason": reason,
    }


def score_one(ticker, sector):
    data = fetch(ticker)
    if data is None:
        return failed_result(ticker, sector, "Brak danych dziennych lub godzinowych")

    weekly, daily, four_hour, hourly = data
    daily_closed = daily.iloc[:-1] if len(daily) > 1 else daily
    weekly_closed = weekly.iloc[:-1] if len(weekly) > 1 else weekly
    h4_closed = four_hour.iloc[:-1] if len(four_hour) > 1 else four_hour

    if len(daily_closed) < 200 or len(weekly_closed) < 20 or len(h4_closed) < 10:
        return failed_result(ticker, sector, "Za mało zamkniętych świec do oceny trendu")

    close = float(daily["Close"].iloc[-1])
    sma50 = float(daily["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(daily["Close"].rolling(200).mean().iloc[-1])
    atr = float((daily["High"] - daily["Low"]).rolling(14).mean().iloc[-1])
    if any(math.isnan(v) for v in [close, sma50, sma200, atr]):
        return failed_result(ticker, sector, "Za mało danych do SMA50/SMA200/ATR")

    weekly_structure = hh_hl(weekly_closed, left=1, right=1)
    weekly_sma20 = float(weekly_closed["Close"].iloc[-1]) > float(
        weekly_closed["Close"].rolling(20).mean().iloc[-1]
    )
    daily_structure = hh_hl(daily_closed, left=2, right=2)
    h4_structure = hh_hl(h4_closed, left=2, right=2)
    weekly_trend = bool(weekly_structure and weekly_sma20)
    daily_trend = bool(daily_structure and close > sma50 and sma50 > sma200)
    return_60d = float(daily["Close"].pct_change(60).iloc[-1])

    pullback = bool(
        daily_trend and abs(close - sma50) <= 1.5 * atr and close > sma200
    )
    previous_high = float(daily_closed["High"].tail(20).max())
    breakout = bool(close > previous_high)
    volume_up = bool(
        len(hourly) >= 20
        and float(hourly["Volume"].tail(5).mean())
        > float(hourly["Volume"].tail(20).mean())
    )

    score = 0
    score += 25 if weekly_trend else 0
    score += 25 if daily_trend else 0
    score += 15 if h4_structure else 0
    score += 15 if return_60d > 0.08 else (8 if return_60d > 0 else 0)
    score += 10 if pullback or breakout else 0
    score += 10 if volume_up else 0

    trend_alignment = bool(weekly_trend and daily_trend and h4_structure)
    qualifies = bool(score >= MIN_SCORE and trend_alignment)
    setup = "Breakout" if breakout else "Pullback" if pullback else "Trend" if h4_structure else "No setup"

    failed = []
    if not weekly_trend:
        failed.append("Trend W1")
    if not daily_trend:
        failed.append("Trend D1 / cena > SMA50 > SMA200")
    if not h4_structure:
        failed.append("Struktura wzrostowa H4")
    if return_60d <= 0:
        failed.append("Dodatni zwrot z 60 sesji")
    if not (pullback or breakout):
        failed.append("Pullback lub Breakout")
    if not volume_up:
        failed.append("Wzrost wolumenu H1 (pomocniczy)")
    if score < MIN_SCORE:
        failed.append(f"Próg {MIN_SCORE} punktów; wynik {score}")
    if score >= MIN_SCORE and not trend_alignment:
        failed.append("Brak pełnej zgodności trendu W1/D1/H4")

    return {
        "ticker": ticker, "sector": sector, "data_ok": True,
        "score": int(score), "qualifies": qualifies,
        "weekly": "HH/HL" if weekly_trend else "Mixed",
        "daily": "HH/HL" if daily_trend else "Mixed",
        "h4_hhhl": h4_structure, "setup4h": setup,
        "entry1h": "Not evaluated", "h1_volume_up": volume_up,
        "close": round(close, 2), "sma50": round(sma50, 2),
        "sma200": round(sma200, 2), "atr14": round(atr, 2),
        "above_sma50": bool(close > sma50),
        "above_sma200": bool(close > sma200),
        "sma50_above_sma200": bool(sma50 > sma200),
        "return_60d_pct": round(return_60d * 100, 2),
        "rs": "A+" if return_60d > 0.15 else "A" if return_60d > 0.08 else "B",
        "pullback": pullback, "breakout": breakout,
        "trend_alignment": trend_alignment, "failed_conditions": failed,
    }


def main():
    universe = pd.read_csv(TICKERS_FILE)
    if not {"ticker", "sector"}.issubset(universe.columns):
        raise ValueError("tickers.csv musi zawierac kolumny: ticker,sector")

    results, errors = [], []
    for _, row in universe.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        sector = str(row["sector"]).strip()
        try:
            result = score_one(ticker, sector)
            results.append(result)
            if not result.get("data_ok", False):
                errors.append({"ticker": ticker, "reason": result.get("reason", "Brak danych")})
        except Exception as error:
            message = str(error)
            print("Pominieto", ticker, message)
            errors.append({"ticker": ticker, "reason": message})
            results.append({
                "ticker": ticker, "sector": sector, "data_ok": False,
                "score": None, "qualifies": False, "reason": message,
            })

    setups = [r for r in results if r.get("data_ok") and r.get("qualifies")]
    setups.sort(key=lambda r: r["score"], reverse=True)
    diagnostics = [r for r in results if r["ticker"] in DIAGNOSTIC_TICKERS]
    diagnostics.sort(key=lambda r: r["ticker"])
    results.sort(key=lambda r: (
        r.get("score") is not None,
        r.get("score") if r.get("score") is not None else -1,
    ), reverse=True)

    output = {
        "updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "universe_count": int(len(universe)),
        "scanned_count": int(sum(1 for r in results if r.get("data_ok"))),
        "qualifying_count": len(setups), "setups": setups[:30],
        "diagnostic_tickers": diagnostics, "all_diagnostics": results,
        "data_errors": errors,
    }
    OUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print(f"Przeskanowano {output['scanned_count']}/{len(universe)} spolek.")
    print(f"Spolki spelniajace prog: {len(setups)}.")
    print("Diagnostyka VST, BWXT, OKLO, CEG:", json.dumps(
        diagnostics, ensure_ascii=False, default=json_default
    ))


if _name_ == "_main_":
    main()
