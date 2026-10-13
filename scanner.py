import json
import math
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# GitHub Actions runs from the repository root; avoid special __file__ syntax.
ROOT = Path(".")
TICKERS_FILE = ROOT / "tickers.csv"
OUT_FILE = ROOT / "data.json"
DIAGNOSTIC_TICKERS = {"VST", "BWXT", "OKLO", "CEG"}
MIN_SCORE = 60
PIVOT_LEFT = 2
PIVOT_RIGHT = 2
PULLBACK_MIN_ATR = 0.5
PULLBACK_MAX_ATR = 2.0
SMA_NEAR_ATR = 0.5
PULLBACK_LOOKBACK_H4 = 30


def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def resample_ohlcv(data, rule):
    return data.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def fetch(ticker):
    stock = yf.Ticker(ticker)
    daily = stock.history(period="2y", interval="1d", auto_adjust=True)
    hourly = stock.history(period="60d", interval="1h", auto_adjust=True)
    if daily.empty or hourly.empty:
        return None
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    weekly = resample_ohlcv(daily, "W-FRI")
    h4 = resample_ohlcv(hourly, "4h")
    return weekly, daily, h4, hourly


def atr_series(data, period=14):
    prev = data["Close"].shift(1)
    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev).abs(),
        (data["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def confirmed_pivots(data, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    high_pivots, low_pivots = [], []
    for i in range(left, len(data) - right):
        hw = highs[i-left:i+right+1]
        lw = lows[i-left:i+right+1]
        if highs[i] == hw.max() and (hw == highs[i]).sum() == 1:
            high_pivots.append((i, float(highs[i])))
        if lows[i] == lw.min() and (lw == lows[i]).sum() == 1:
            low_pivots.append((i, float(lows[i])))
    return high_pivots, low_pivots


def rising_structure(data, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    highs, lows = confirmed_pivots(data, left, right)
    return bool(
        len(highs) >= 2 and len(lows) >= 2
        and highs[-1][1] > highs[-2][1]
        and lows[-1][1] > lows[-2][1]
    )


def failed_result(ticker, sector, reason):
    return {
        "ticker": ticker, "sector": sector, "data_ok": False,
        "score": None, "qualifies": False, "status": "Data error",
        "reason": reason,
    }


def h4_setup(h4):
    empty = {
        "pullback": False, "breakout_confirmed": False,
        "pivot_high": None, "pullback_depth_atr": None,
        "near_sma50": False,
    }
    if len(h4) < 65:
        return empty

    # Ignore the newest H4 bucket because it may still be forming.
    closed = h4.iloc[:-1].copy()
    if len(closed) < 60:
        return empty
    closed["SMA50"] = closed["Close"].rolling(50).mean()
    closed["ATR14"] = atr_series(closed, 14)
    last_close = float(closed["Close"].iloc[-1])
    last_atr = float(closed["ATR14"].iloc[-1])
    if math.isnan(last_atr) or last_atr <= 0:
        return empty

    pivots, _ = confirmed_pivots(closed)
    eligible = [(i, price) for i, price in pivots if i < len(closed) - 1]
    if not eligible:
        return empty
    pivot_i, pivot_price = eligible[-1]
    after = closed.iloc[pivot_i + 1:]
    if after.empty:
        return empty

    recent = after.tail(PULLBACK_LOOKBACK_H4)
    lowest = float(recent["Low"].min())
    depth = max(0.0, (pivot_price - lowest) / last_atr)
    valid = recent["SMA50"].notna()
    near_sma = False
    if valid.any():
        dist = (recent.loc[valid, "Close"] - recent.loc[valid, "SMA50"]).abs()
        near_sma = bool((dist <= SMA_NEAR_ATR * last_atr).any())

    pullback = bool(near_sma or PULLBACK_MIN_ATR <= depth <= PULLBACK_MAX_ATR)
    # Breakout must be above a pivot confirmed before the final closed H4 bar.
    breakout = bool(pullback and last_close > pivot_price)
    return {
        "pullback": pullback,
        "breakout_confirmed": breakout,
        "pivot_high": round(pivot_price, 2),
        "pullback_depth_atr": round(depth, 2),
        "near_sma50": near_sma,
    }


def score_one(ticker, sector):
    data = fetch(ticker)
    if data is None:
        return failed_result(ticker, sector, "Brak danych dziennych lub godzinowych")
    weekly, daily, h4, hourly = data

    # Trend filters use completed daily and weekly bars only.
    d = daily.iloc[:-1].copy() if len(daily) > 1 else daily.copy()
    w = weekly.iloc[:-1].copy() if len(weekly) > 1 else weekly.copy()
    h4_closed = h4.iloc[:-1].copy() if len(h4) > 1 else h4.copy()
    if len(d) < 200 or len(w) < 20 or len(h4_closed) < 60:
        return failed_result(ticker, sector, "Za malo danych do srednich i struktury")

    close = float(d["Close"].iloc[-1])
    sma50 = float(d["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(d["Close"].rolling(200).mean().iloc[-1])
    sma20w = float(w["Close"].rolling(20).mean().iloc[-1])
    if any(math.isnan(v) for v in [close, sma50, sma200, sma20w]):
        return failed_result(ticker, sector, "Za malo danych do obliczenia srednich")

    weekly_trend = bool(
        rising_structure(w, left=1, right=1) and float(w["Close"].iloc[-1]) > sma20w
    )
    daily_trend = bool(rising_structure(d) and close > sma50 and sma50 > sma200)
    h4_trend = rising_structure(h4_closed)
    setup = h4_setup(h4)

    ret60 = float(d["Close"].pct_change(60).iloc[-1])
    volume_up = bool(
        len(hourly) >= 20
        and float(hourly["Volume"].tail(5).mean())
        > float(hourly["Volume"].tail(20).mean())
    )
    aligned = bool(weekly_trend and daily_trend and h4_trend)

    if not weekly_trend or not daily_trend:
        status = "Trend invalidated"
    elif setup["breakout_confirmed"] and aligned:
        status = "Confirmed setup"
    elif setup["pullback"]:
        status = "Pullback waiting"
    else:
        status = "Trend OK"

    score = 0
    score += 25 if weekly_trend else 0
    score += 25 if daily_trend else 0
    score += 15 if h4_trend else 0
    score += 15 if ret60 > 0.08 else (8 if ret60 > 0 else 0)
    score += 10 if setup["pullback"] else 0
    score += 10 if volume_up else 0
    qualifies = bool(status == "Confirmed setup" and score >= MIN_SCORE)

    failed = []
    if not weekly_trend:
        failed.append("W1: HH/HL i zamkniecie powyzej SMA20 W1")
    if not daily_trend:
        failed.append("D1: HH/HL i cena > SMA50 > SMA200")
    if not h4_trend:
        failed.append("H4: wzrostowa struktura HH/HL")
    if not setup["pullback"]:
        failed.append("H4: brak korekty przy SMA50 lub cofniecia 0,5-2,0 ATR")
    elif not setup["breakout_confirmed"]:
        failed.append("H4: oczekiwanie na zamkniecie powyzej pivot high")
    if ret60 <= 0:
        failed.append("Dodatni zwrot z 60 sesji")
    if score < MIN_SCORE:
        failed.append("Prog punktowy " + str(MIN_SCORE) + "; wynik " + str(score))

    h4sma50 = float(h4_closed["Close"].rolling(50).mean().iloc[-1])
    h4atr = float(atr_series(h4_closed, 14).iloc[-1])
    datr = float(atr_series(d, 14).iloc[-1])
    return {
        "ticker": ticker, "sector": sector, "data_ok": True,
        "score": int(score), "qualifies": qualifies, "status": status,
        "weekly": "HH/HL" if weekly_trend else "Mixed",
        "daily": "HH/HL" if daily_trend else "Mixed",
        "h4_hhhl": bool(h4_trend), "setup4h": status,
        "entry1h": "Not evaluated", "h1_volume_up": volume_up,
        "close": round(close, 2), "sma50": round(sma50, 2),
        "sma200": round(sma200, 2), "sma20_weekly": round(sma20w, 2),
        "sma50_h4": round(h4sma50, 2) if not math.isnan(h4sma50) else None,
        "atr14": round(datr, 2) if not math.isnan(datr) else None,
        "atr14_h4": round(h4atr, 2) if not math.isnan(h4atr) else None,
        "above_sma50": bool(close > sma50),
        "above_sma200": bool(close > sma200),
        "sma50_above_sma200": bool(sma50 > sma200),
        "return_60d_pct": round(ret60 * 100, 2),
        "rs": "A+" if ret60 > 0.15 else "A" if ret60 > 0.08 else "B",
        "pullback": bool(setup["pullback"]),
        "breakout": bool(setup["breakout_confirmed"]),
        "pivot_high_h4": setup["pivot_high"],
        "pullback_depth_atr": setup["pullback_depth_atr"],
        "near_sma50_h4": setup["near_sma50"],
        "trend_alignment": aligned, "failed_conditions": failed,
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
            print("Pominieto " + ticker + ": " + message)
            errors.append({"ticker": ticker, "reason": message})
            results.append({
                "ticker": ticker, "sector": sector, "data_ok": False,
                "score": None, "qualifies": False, "status": "Data error",
                "reason": message,
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
        "qualifying_count": len(setups),
        "setups": setups[:30],
        "diagnostic_tickers": diagnostics,
        "all_diagnostics": results,
        "data_errors": errors,
    }
    OUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print("Przeskanowano " + str(output["scanned_count"]) + "/" + str(len(universe)) + " spolek.")
    print("Potwierdzone setupy: " + str(len(setups)) + ".")
    print("Diagnostyka: " + json.dumps(diagnostics, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
