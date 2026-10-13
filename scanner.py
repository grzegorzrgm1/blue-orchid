import json
import math
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# GitHub Actions starts in the repository root; avoid the special __file__ variable.
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


def now_for_index(index):
    """Get current time in the data index timezone when possible."""
    try:
        if index.tz is not None:
            return pd.Timestamp.now(tz=index.tz)
    except (AttributeError, TypeError):
        pass
    return pd.Timestamp.now()


def resample_ohlcv(data, rule, offset=None):
    kwargs = {"label": "left", "closed": "left"}
    if offset is not None:
        kwargs["origin"] = "start_day"
        kwargs["offset"] = offset
    return data.resample(rule, **kwargs).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def completed_daily_bars(daily):
    """Drop today's daily candle only while the US regular session may be open."""
    if daily.empty:
        return daily.copy()
    now = now_for_index(daily.index)
    latest = daily.index[-1]
    if latest.date() == now.date() and (now.hour, now.minute) < (16, 5):
        return daily.iloc[:-1].copy()
    return daily.copy()


def completed_weekly_bars(weekly):
    """Drop the current incomplete weekly bucket, not the prior completed week."""
    if weekly.empty:
        return weekly.copy()
    now = now_for_index(weekly.index)
    latest_date = weekly.index[-1].date()
    if latest_date > now.date():
        return weekly.iloc[:-1].copy()
    if latest_date == now.date() and (now.hour, now.minute) < (16, 5):
        return weekly.iloc[:-1].copy()
    return weekly.copy()


def completed_h4_bars(h4):
    """Exclude a current-day H4 bucket until its expected session close."""
    if h4.empty:
        return h4.copy()
    now = now_for_index(h4.index)
    latest = h4.index[-1]
    if latest.date() < now.date():
        return h4.copy()
    if latest.date() > now.date():
        return h4.iloc[:-1].copy()

    # H4 is anchored to 09:30 US market time: first bucket ends at 13:30,
    # and the shorter final regular-session bucket ends at 16:00.
    bar_minutes = latest.hour * 60 + latest.minute
    expected_close = (13 * 60 + 30) if bar_minutes < (13 * 60 + 30) else (16 * 60 + 5)
    now_minutes = now.hour * 60 + now.minute
    if now_minutes < expected_close:
        return h4.iloc[:-1].copy()
    return h4.copy()


def atr_series(data, period=14):
    previous_close = data["Close"].shift(1)
    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def fetch(ticker):
    stock = yf.Ticker(ticker)
    daily_raw = stock.history(
        period="2y", interval="1d", auto_adjust=True, prepost=False
    )
    hourly = stock.history(
        period="60d", interval="1h", auto_adjust=True, prepost=False
    )
    if daily_raw.empty or hourly.empty:
        return None

    daily_raw = daily_raw[~daily_raw.index.duplicated(keep="last")].sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    daily_raw = completed_daily_bars(daily_raw)
    if daily_raw.empty:
        return None

    weekly_raw = resample_ohlcv(daily_raw, "W-FRI")
    weekly = completed_weekly_bars(weekly_raw)

    # Anchor H4 buckets at the US regular-session open rather than midnight.
    h4_raw = resample_ohlcv(hourly, "4h", offset="9h30min")
    h4 = completed_h4_bars(h4_raw)
    return weekly, daily_raw, h4, hourly


def confirmed_pivots(data, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    """Return local pivots only after all requested right-side bars exist."""
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    high_pivots = []
    low_pivots = []
    for i in range(left, len(data) - right):
        high_window = highs[i-left:i+right+1]
        low_window = lows[i-left:i+right+1]
        if highs[i] == high_window.max() and (high_window == highs[i]).sum() == 1:
            high_pivots.append((i, float(highs[i])))
        if lows[i] == low_window.min() and (low_window == lows[i]).sum() == 1:
            low_pivots.append((i, float(lows[i])))
    return high_pivots, low_pivots


def rising_structure(data, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    highs, lows = confirmed_pivots(data, left, right)
    if len(highs) < 2 or len(lows) < 2:
        return False
    return bool(
        highs[-1][1] > highs[-2][1]
        and lows[-1][1] > lows[-2][1]
    )


def falling_structure(data, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    highs, lows = confirmed_pivots(data, left, right)
    if len(highs) < 2 or len(lows) < 2:
        return False
    return bool(
        highs[-1][1] < highs[-2][1]
        and lows[-1][1] < lows[-2][1]
    )


def failed_result(ticker, sector, reason):
    return {
        "ticker": ticker,
        "sector": sector,
        "data_ok": False,
        "score": None,
        "qualifies": False,
        "status": "Data error",
        "reason": reason,
    }


def h4_setup(h4):
    """Detect an H4 pullback followed by a closed-candle pivot-high breakout."""
    empty = {
        "pullback": False,
        "breakout_confirmed": False,
        "pivot_high": None,
        "pullback_depth_atr": None,
        "near_sma50": False,
    }
    closed = completed_h4_bars(h4)
    if len(closed) < 65:
        return empty

    closed = closed.copy()
    closed["SMA50"] = closed["Close"].rolling(50).mean()
    closed["ATR14"] = atr_series(closed, 14)
    last_close = float(closed["Close"].iloc[-1])
    last_atr = float(closed["ATR14"].iloc[-1])
    if math.isnan(last_atr) or last_atr <= 0:
        return empty

    high_pivots, _ = confirmed_pivots(closed)
    # The pivot level must have been confirmed before the last closed H4 candle.
    eligible = [(i, price) for i, price in high_pivots if i < len(closed) - 1]
    if not eligible:
        return empty

    pivot_index, pivot_price = eligible[-1]
    after_pivot = closed.iloc[pivot_index + 1:]
    if after_pivot.empty:
        return empty

    recent = after_pivot.tail(PULLBACK_LOOKBACK_H4)
    lowest_low = float(recent["Low"].min())
    depth_atr = max(0.0, (pivot_price - lowest_low) / last_atr)

    near_sma50 = False
    valid_sma = recent["SMA50"].notna() & recent["ATR14"].notna()
    if valid_sma.any():
        relevant = recent.loc[valid_sma]
        sma = relevant["SMA50"]
        bar_atr = relevant["ATR14"]
        distance = pd.concat([
            (relevant["Low"] - sma).abs(),
            (relevant["High"] - sma).abs(),
        ], axis=1).min(axis=1)
        crosses_sma = (relevant["Low"] <= sma) & (relevant["High"] >= sma)
        near_sma50 = bool((crosses_sma | (distance <= SMA_NEAR_ATR * bar_atr)).any())

    depth_valid = PULLBACK_MIN_ATR <= depth_atr <= PULLBACK_MAX_ATR
    pullback = bool(depth_atr <= PULLBACK_MAX_ATR and (near_sma50 or depth_valid))
    breakout_confirmed = bool(pullback and last_close > pivot_price)

    return {
        "pullback": pullback,
        "breakout_confirmed": breakout_confirmed,
        "pivot_high": round(pivot_price, 2),
        "pullback_depth_atr": round(depth_atr, 2),
        "near_sma50": near_sma50,
    }



def score_one(ticker, sector):
    data = fetch(ticker)
    if data is None:
        return failed_result(ticker, sector, "Brak danych dziennych lub godzinowych")

    weekly, daily, h4, hourly = data
    if len(daily) < 200 or len(weekly) < 20 or len(h4) < 60:
        return failed_result(ticker, sector, "Za malo danych do srednich i struktury")

    close = float(daily["Close"].iloc[-1])
    sma50_d = float(daily["Close"].rolling(50).mean().iloc[-1])
    sma200_d = float(daily["Close"].rolling(200).mean().iloc[-1])
    sma20_w = float(weekly["Close"].rolling(20).mean().iloc[-1])
    if any(math.isnan(value) for value in [close, sma50_d, sma200_d, sma20_w]):
        return failed_result(ticker, sector, "Za malo danych do obliczenia srednich")

    weekly_trend = bool(
        rising_structure(weekly)
        and float(weekly["Close"].iloc[-1]) > sma20_w
    )
    daily_trend = bool(
        rising_structure(daily)
        and close > sma50_d
        and sma50_d > sma200_d
    )

    h4_trend = rising_structure(h4)
    h4_downtrend = falling_structure(h4)
    setup = h4_setup(h4)

    return_60d = float(daily["Close"].pct_change(60).iloc[-1])
    h1_volume_up = bool(
        len(hourly) >= 20
        and float(hourly["Volume"].tail(5).mean())
        > float(hourly["Volume"].tail(20).mean())
    )

    # The pullback may interrupt the H4 HH/HL sequence, so H4 structure is
    # diagnostic context. The required directional gate is W1 + D1.
    higher_timeframe_trend = bool(weekly_trend and daily_trend)
    confirmed_setup = bool(
        higher_timeframe_trend
        and not h4_downtrend
        and setup["breakout_confirmed"]
    )

    if not higher_timeframe_trend or h4_downtrend:
        status = "Trend invalidated"
    elif confirmed_setup:
        status = "Confirmed setup"
    elif setup["pullback"]:
        status = "Pullback waiting"
    else:
        status = "Trend OK"

    score = 0
    score += 25 if weekly_trend else 0
    score += 25 if daily_trend else 0
    score += 15 if h4_trend else 0
    score += 15 if return_60d > 0.08 else (8 if return_60d > 0 else 0)
    score += 10 if setup["pullback"] else 0
    score += 10 if h1_volume_up else 0
    qualifies = bool(status == "Confirmed setup" and score >= MIN_SCORE)

    failed = []
    if not weekly_trend:
        failed.append("W1: HH/HL i zamkniecie powyzej SMA20 W1")
    if not daily_trend:
        failed.append("D1: HH/HL i cena > SMA50 > SMA200")
    if h4_downtrend:
        failed.append("H4: potwierdzona struktura spadkowa")
    if not setup["pullback"]:
        failed.append("H4: brak korekty przy SMA50 lub cofniecia 0,5-2,0 ATR")
    elif not setup["breakout_confirmed"]:
        failed.append("H4: oczekiwanie na zamkniecie powyzej potwierdzonego pivot high 2+2")
    if return_60d <= 0:
        failed.append("Dodatni zwrot z 60 sesji")
    if score < MIN_SCORE:
        failed.append("Prog punktowy " + str(MIN_SCORE) + "; wynik " + str(score))

    h4_sma50 = float(h4["Close"].rolling(50).mean().iloc[-1])
    h4_atr = float(atr_series(h4, 14).iloc[-1])
    daily_atr = float(atr_series(daily, 14).iloc[-1])

    return {
        "ticker": ticker,
        "sector": sector,
        "data_ok": True,
        "score": int(score),
        "qualifies": qualifies,
        "status": status,
        "weekly": "HH/HL" if weekly_trend else "Mixed",
        "daily": "HH/HL" if daily_trend else "Mixed",
        "h4_hhhl": bool(h4_trend),
        "setup4h": status,
        "entry1h": "Not evaluated",
        "h1_volume_up": h1_volume_up,
        "close": round(close, 2),
        "sma50": round(sma50_d, 2),
        "sma200": round(sma200_d, 2),
        "sma20_weekly": round(sma20_w, 2),
        "sma50_h4": round(h4_sma50, 2) if not math.isnan(h4_sma50) else None,
        "atr14": round(daily_atr, 2) if not math.isnan(daily_atr) else None,
        "atr14_h4": round(h4_atr, 2) if not math.isnan(h4_atr) else None,
        "above_sma50": bool(close > sma50_d),
        "above_sma200": bool(close > sma200_d),
        "sma50_above_sma200": bool(sma50_d > sma200_d),
        "return_60d_pct": round(return_60d * 100, 2),
        "rs": "A+" if return_60d > 0.15 else "A" if return_60d > 0.08 else "B",
        "pullback": bool(setup["pullback"]),
        "breakout": bool(setup["breakout_confirmed"]),
        "pivot_high_h4": setup["pivot_high"],
        "pullback_depth_atr": setup["pullback_depth_atr"],
        "near_sma50_h4": setup["near_sma50"],
        "trend_alignment": higher_timeframe_trend,
        "failed_conditions": failed,
    }


def main():
    universe = pd.read_csv(TICKERS_FILE)
    if not {"ticker", "sector"}.issubset(universe.columns):
        raise ValueError("tickers.csv musi zawierac kolumny: ticker,sector")

    results = []
    errors = []
    for _, row in universe.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        sector = str(row["sector"]).strip()
        try:
            result = score_one(ticker, sector)
            results.append(result)
            if not result.get("data_ok", False):
                errors.append({
                    "ticker": ticker,
                    "reason": result.get("reason", "Brak danych"),
                })
        except Exception as error:
            message = str(error)
            print("Pominieto " + ticker + ": " + message)
            errors.append({"ticker": ticker, "reason": message})
            results.append({
                "ticker": ticker,
                "sector": sector,
                "data_ok": False,
                "score": None,
                "qualifies": False,
                "status": "Data error",
                "reason": message,
            })

    setups = [
        result for result in results
        if result.get("data_ok") and result.get("qualifies")
    ]
    setups.sort(key=lambda result: result["score"], reverse=True)
    diagnostics = [
        result for result in results
        if result["ticker"] in DIAGNOSTIC_TICKERS
    ]
    diagnostics.sort(key=lambda result: result["ticker"])
    results.sort(key=lambda result: (
        result.get("score") is not None,
        result.get("score") if result.get("score") is not None else -1,
    ), reverse=True)

    output = {
        "updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "universe_count": int(len(universe)),
        "scanned_count": int(sum(1 for result in results if result.get("data_ok"))),
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
