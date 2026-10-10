import json
import math
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(_file_).resolve().parent
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

    daily = stock.history(
        period="2y",
        interval="1d",
        auto_adjust=True,
    )
    hourly = stock.history(
        period="60d",
        interval="1h",
        auto_adjust=True,
    )

    if daily.empty or hourly.empty:
        return None

    four_hour = hourly.resample("4h").agg({
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

    return weekly, daily, four_hour, hourly


def hh_hl(data, lookback=5):
    if len(data) < lookback * 2:
        return False

    recent = data.tail(lookback)
    previous = data.iloc[-lookback * 2:-lookback]

    return bool(
        recent["High"].max() >= previous["High"].max()
        and recent["Low"].min() >= previous["Low"].min()
    )


def failed_result(ticker, sector, reason):
    return {
        "ticker": ticker,
        "sector": sector,
        "data_ok": False,
        "score": None,
        "qualifies": False,
        "reason": reason,
    }


def score_one(ticker, sector):
    data = fetch(ticker)

    if data is None:
        return failed_result(
            ticker,
            sector,
            "Brak danych dziennych lub godzinowych",
        )

    weekly, daily, four_hour, hourly = data

    close = float(daily["Close"].iloc[-1])
    sma50 = float(daily["Close"].rolling(50).mean().iloc[-1])
    sma200 = float(daily["Close"].rolling(200).mean().iloc[-1])
    atr = float(
        (daily["High"] - daily["Low"]).rolling(14).mean().iloc[-1]
    )

    if any(math.isnan(value) for value in [sma50, sma200, atr]):
        return failed_result(
            ticker,
            sector,
            "Za malo danych do obliczenia SMA50/SMA200/ATR",
        )

    weekly_hhhl = hh_hl(weekly)
    weekly_sma20 = (
        len(weekly) >= 20
        and float(weekly["Close"].iloc[-1])
        > float(weekly["Close"].rolling(20).mean().iloc[-1])
    )

    daily_hhhl = hh_hl(daily)
    four_hour_hhhl = hh_hl(four_hour)

    weekly_trend = bool(weekly_hhhl and weekly_sma20)
    daily_trend = bool(
        daily_hhhl and close > sma50 and sma50 > sma200
    )

    # Zwrot ceny z ostatnich 60 sesji.
    # To jeszcze nie jest relatywna sila wobec indeksu.
    return_60d = float(daily["Close"].pct_change(60).iloc[-1])

    pullback = bool(
        abs(close - sma50) <= 2.0 * atr
        and close > sma200
    )

    previous_high = float(
        daily["High"].tail(20).iloc[:-1].max()
    )
    breakout = bool(close >= previous_high)

    volume_up = bool(
        len(hourly) >= 20
        and float(hourly["Volume"].tail(5).mean())
        > float(hourly["Volume"].tail(20).mean())
    )

    score = 0
    score += 25 if weekly_trend else 0
    score += 25 if daily_trend else 0
    score += 15 if four_hour_hhhl else 0
    score += 15 if return_60d > 0.08 else (
        8 if return_60d > 0 else 0
    )
    score += 10 if pullback or breakout else 0
    score += 10 if volume_up else 0

    setup = (
        "Breakout"
        if breakout
        else "Pullback"
        if pullback
        else "Trend"
    )

    qualifies = score >= MIN_SCORE

    failed = []

    if not weekly_trend:
        failed.append("Trend W1")
    if not daily_trend:
        failed.append("Trend D1 / cena > SMA50 > SMA200")
    if not four_hour_hhhl:
        failed.append("Układ HH/HL H4")
    if return_60d <= 0:
        failed.append("Dodatni zwrot z 60 sesji")
    if not (pullback or breakout):
        failed.append("Pullback lub Breakout")
    if not volume_up:
        failed.append("Wzrost wolumenu H1")
    if not qualifies:
        failed.append(f"Próg {MIN_SCORE} punktów; wynik {score}")

    return {
        "ticker": ticker,
        "sector": sector,
        "data_ok": True,
        "score": int(score),
        "qualifies": qualifies,
        "weekly": "HH/HL" if weekly_trend else "Mixed",
        "daily": "HH/HL" if daily_trend else "Mixed",
        "h4_hhhl": four_hour_hhhl,
        "setup4h": setup,
        "entry1h": "Confirm" if four_hour_hhhl else "Watch",
        "h1_volume_up": volume_up,
        "close": round(close, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "atr14": round(atr, 2),
        "above_sma50": bool(close > sma50),
        "above_sma200": bool(close > sma200),
        "sma50_above_sma200": bool(sma50 > sma200),
        "return_60d_pct": round(return_60d * 100, 2),
        "rs": (
            "A+"
            if return_60d > 0.15
            else "A"
            if return_60d > 0.08
            else "B"
        ),
        "pullback": pullback,
        "breakout": breakout,
        "failed_conditions": failed,
    }


def main():
    universe = pd.read_csv(TICKERS_FILE)

    required_columns = {"ticker", "sector"}
    if not required_columns.issubset(universe.columns):
        raise ValueError(
            "tickers.csv musi zawierac kolumny: ticker,sector"
        )

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
            print("Pominieto", ticker, message)

            errors.append({
                "ticker": ticker,
                "reason": message,
            })

            results.append({
                "ticker": ticker,
                "sector": sector,
                "data_ok": False,
                "score": None,
                "qualifies": False,
                "reason": message,
            })

    setups = [
        result
        for result in results
        if result.get("data_ok") and result.get("qualifies")
    ]
    setups.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    diagnostics = [
        result
        for result in results
        if result["ticker"] in DIAGNOSTIC_TICKERS
    ]
    diagnostics.sort(key=lambda result: result["ticker"])

    results.sort(
        key=lambda result: (
            result.get("score") is not None,
            result.get("score") if result.get("score") is not None else -1,
        ),
        reverse=True,
    )

    output = {
        "updated": time.strftime(
            "%Y-%m-%d %H:%M UTC",
            time.gmtime(),
        ),
        "universe_count": int(len(universe)),
        "scanned_count": int(
            sum(1 for result in results if result.get("data_ok"))
        ),
        "qualifying_count": len(setups),
        "setups": setups[:30],
        "diagnostic_tickers": diagnostics,
        "all_diagnostics": results,
        "data_errors": errors,
    }

    OUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print(
        f"Przeskanowano {output['scanned_count']}/"
        f"{len(universe)} spolek."
    )
    print(
        f"Spolki spelniajace prog: {len(setups)}."
    )
    print(
        "Diagnostyka VST, BWXT, OKLO, CEG:",
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            default=json_default,
        ),
    )


if _name_ == "_main_":
    main()
