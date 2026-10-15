import json
import math
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# GitHub Actions runs from the repository root.
ROOT = Path(".")
TICKERS_FILE = ROOT / "tickers.csv"
OUT_FILE = ROOT / "data.json"

DIAGNOSTIC_TICKERS = {"VST", "BWXT", "OKLO", "CEG"}
MIN_SCORE = 60

EMA_FAST = 20
EMA_SLOW = 50
PULLBACK_ATR_TOLERANCE = 0.5
PULLBACK_LOOKBACK_H4 = 8


def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def now_for_index(index):
    try:
        timezone = index.tz
        if timezone is not None:
            return pd.Timestamp.now(tz=timezone)
    except (AttributeError, TypeError):
        pass
    return pd.Timestamp.now()


def resample_ohlcv(data, rule, offset=None, label="left", closed="left"):
    options = {"label": label, "closed": closed}
    if offset is not None:
        options["origin"] = "start_day"
        options["offset"] = offset
    return data.resample(rule, **options).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()


def completed_daily_bars(daily):
    if daily.empty:
        return daily.copy()
    now = now_for_index(daily.index)
    latest = daily.index[-1]
    if latest.date() == now.date() and (now.hour, now.minute) < (16, 5):
        return daily.iloc[:-1].copy()
    return daily.copy()


def completed_weekly_bars(weekly):
    if weekly.empty:
        return weekly.copy()
    now = now_for_index(weekly.index)
    latest = weekly.index[-1]
    if latest.date() > now.date():
        return weekly.iloc[:-1].copy()
    if latest.date() == now.date() and (now.hour, now.minute) < (16, 5):
        return weekly.iloc[:-1].copy()
    return weekly.copy()


def completed_h4_bars(h4):
    if h4.empty:
        return h4.copy()
    now = now_for_index(h4.index)
    latest = h4.index[-1]

    if latest.date() < now.date():
        return h4.copy()
    if latest.date() > now.date():
        return h4.iloc[:-1].copy()

    # Regular US session H4 buckets start at 09:30 and 13:30 New York time.
    # The second bucket is shorter because the regular session closes at 16:00.
    start_minutes = latest.hour * 60 + latest.minute
    expected_close = (13 * 60 + 30) if start_minutes < (13 * 60 + 30) else (16 * 60 + 5)
    now_minutes = now.hour * 60 + now.minute

    if now_minutes < expected_close:
        return h4.iloc[:-1].copy()
    return h4.copy()


def ema_series(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


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
    # Yahoo does not provide a dependable native 4-hour interval through yfinance.
    # Use 30-minute bars only as raw input to construct H4 candles.
    intraday_30m = stock.history(
        period="60d", interval="30m", auto_adjust=True, prepost=False
    )

    if daily_raw.empty or intraday_30m.empty:
        return None

    daily_raw = daily_raw[~daily_raw.index.duplicated(keep="last")].sort_index()
    intraday_30m = intraday_30m[
        ~intraday_30m.index.duplicated(keep="last")
    ].sort_index()

    daily = completed_daily_bars(daily_raw)
    if daily.empty:
        return None

    weekly = resample_ohlcv(daily, "W-FRI", label="right", closed="right")
    weekly = completed_weekly_bars(weekly)

    # Group 30-minute data into regular-session H4 bars, anchored at 09:30 ET.
    h4 = resample_ohlcv(intraday_30m, "4h", offset="9h30min")
    h4 = completed_h4_bars(h4)

    return weekly, daily, h4


def directional_trend(close, ema20, ema50, direction):
    if any(pd.isna(value) for value in [close, ema20, ema50]):
        return False
    if direction == "LONG":
        return bool(close > ema20 > ema50)
    return bool(close < ema20 < ema50)


def pullback_near_emas(h4, direction):
    """Look for a recent pullback BEFORE the latest closed H4 candle."""
    result = {
        "pullback": False,
        "near_ema20": False,
        "near_ema50": False,
        "pullback_atr": None,
    }
    if len(h4) < 65:
        return result

    data = h4.copy()
    data["EMA20"] = ema_series(data["Close"], EMA_FAST)
    data["EMA50"] = ema_series(data["Close"], EMA_SLOW)
    data["ATR14"] = atr_series(data, 14)

    # Exclude the latest closed candle: it is evaluated separately as the bounce.
    recent = data.iloc[:-1].tail(PULLBACK_LOOKBACK_H4)
    recent = recent.dropna(subset=["EMA20", "EMA50", "ATR14"])
    if recent.empty:
        return result

    near20_values = []
    near50_values = []
    depths = []
    for _, candle in recent.iterrows():
        atr = float(candle["ATR14"])
        if atr <= 0:
            continue
        tolerance = PULLBACK_ATR_TOLERANCE * atr
        low = float(candle["Low"])
        high = float(candle["High"])
        ema20 = float(candle["EMA20"])
        ema50 = float(candle["EMA50"])
        near20_values.append((low <= ema20 + tolerance) and (high >= ema20 - tolerance))
        near50_values.append((low <= ema50 + tolerance) and (high >= ema50 - tolerance))
        if direction == "LONG":
            depths.append(max(0.0, (ema20 - low) / atr))
        else:
            depths.append(max(0.0, (high - ema20) / atr))

    near20 = any(near20_values)
    near50 = any(near50_values)
    depth = max(depths) if depths else None
    result.update({
        "pullback": bool(near20 or near50),
        "near_ema20": bool(near20),
        "near_ema50": bool(near50),
        "pullback_atr": round(float(depth), 2) if depth is not None else None,
    })
    return result


def bounce_confirmed(h4, direction):
    """Require the latest closed H4 candle to turn back through EMA20."""
    if len(h4) < EMA_SLOW + 2:
        return False
    close = h4["Close"]
    ema20 = ema_series(close, EMA_FAST)
    last = h4.iloc[-1]
    last_ema20 = float(ema20.iloc[-1])
    last_close = float(last["Close"])
    last_open = float(last["Open"])
    if direction == "LONG":
        return bool(last_close > last_ema20 and last_close > last_open)
    return bool(last_close < last_ema20 and last_close < last_open)


def evaluate_direction(ticker, sector, direction, weekly, daily, h4, return_60d):
    if len(weekly) < EMA_SLOW or len(daily) < EMA_SLOW or len(h4) < 65:
        return {
            "ticker": ticker, "sector": sector, "direction": direction,
            "data_ok": True, "score": 0, "qualifies": False,
            "status": "Trend invalidated",
            "reason": "Za malo danych do EMA20/EMA50 lub H4",
            "failed_conditions": ["Za malo danych do obliczen EMA"],
        }

    weekly_close = float(weekly["Close"].iloc[-1])
    daily_close = float(daily["Close"].iloc[-1])
    ema20_w = float(ema_series(weekly["Close"], EMA_FAST).iloc[-1])
    ema50_w = float(ema_series(weekly["Close"], EMA_SLOW).iloc[-1])
    ema20_d_series = ema_series(daily["Close"], EMA_FAST)
    ema50_d_series = ema_series(daily["Close"], EMA_SLOW)
    ema20_d = float(ema20_d_series.iloc[-1])
    ema50_d = float(ema50_d_series.iloc[-1])

    h4_closed = completed_h4_bars(h4)
    if len(h4_closed) < 65:
        return {
            "ticker": ticker, "sector": sector, "direction": direction,
            "data_ok": True, "score": 0, "qualifies": False,
            "status": "Trend invalidated",
            "reason": "Za malo zamknietych swiec H4",
            "failed_conditions": ["Za malo danych do obliczen EMA"],
        }

    h4_close = float(h4_closed["Close"].iloc[-1])
    ema20_h4_series = ema_series(h4_closed["Close"], EMA_FAST)
    ema50_h4_series = ema_series(h4_closed["Close"], EMA_SLOW)
    ema20_h4 = float(ema20_h4_series.iloc[-1])
    ema50_h4 = float(ema50_h4_series.iloc[-1])

    weekly_trend = directional_trend(weekly_close, ema20_w, ema50_w, direction)
    daily_trend = directional_trend(daily_close, ema20_d, ema50_d, direction)
    if direction == "LONG":
        h4_invalidated = bool(h4_close < ema50_h4)
        directional_return = return_60d
    else:
        h4_invalidated = bool(h4_close > ema50_h4)
        directional_return = -return_60d

    pullback = pullback_near_emas(h4_closed, direction)
    bounce = bounce_confirmed(h4_closed, direction)
    higher_timeframe_trend = bool(weekly_trend and daily_trend)

    # Avoid ranking a stretched stock as an attractive fresh setup.
    daily_atr = float(atr_series(daily, 14).iloc[-1])
    if daily_atr > 0:
        extension_atr = ((daily_close - ema20_d) / daily_atr) if direction == "LONG" else ((ema20_d - daily_close) / daily_atr)
    else:
        extension_atr = 999.0
    extended = bool(extension_atr > 2.0)

    if not higher_timeframe_trend or h4_invalidated:
        status = "Trend invalidated"
    elif pullback["pullback"] and bounce and not extended:
        status = "Confirmed setup"
    elif pullback["pullback"] and not bounce:
        status = "Pullback waiting"
    else:
        # Trend remains valid, but no fresh, non-extended pullback/bounce is present.
        status = "Trend OK"

    # Score now rewards a complete setup, not just a strong trend.
    score = 0
    score += 20 if weekly_trend else 0
    score += 25 if daily_trend else 0
    score += 15 if not h4_invalidated else 0
    score += 15 if pullback["pullback"] else 0
    score += 15 if bounce and pullback["pullback"] else 0
    score += 10 if not extended else 0
    score = min(100, int(score))

    h4_distance_pct = abs(h4_close - ema20_h4) / h4_close * 100.0 if h4_close else None
    qualifies = bool(status == "Confirmed setup" and score >= MIN_SCORE)
    failed = []
    if not weekly_trend:
        failed.append("W1: cena nie jest po wlasciwej stronie EMA20 i EMA50")
    if not daily_trend:
        failed.append("D1: cena nie jest po wlasciwej stronie EMA20 i EMA50")
    if h4_invalidated:
        failed.append("H4: zamkniecie po niewlasciwej stronie EMA50")
    elif not pullback["pullback"]:
        failed.append("H4: brak korekty w okolice EMA20/EMA50 w ostatnich swiecach")
    elif not bounce:
        failed.append("H4: korekta byla, ale ostatnia zamknieta swieca nie potwierdzila odbicia przez EMA20")
    if extended:
        failed.append("D1: cena jest ponad 2 ATR od EMA20 — trend jest rozciagniety, nie traktuj jako swiezego setupu")
    if score < MIN_SCORE:
        failed.append("Ponizej progu punktowego " + str(MIN_SCORE))

    return {
        "ticker": ticker, "sector": sector, "direction": direction,
        "data_ok": True, "score": score, "qualifies": qualifies,
        "status": status,
        "weekly": "Above EMA20/EMA50" if weekly_trend and direction == "LONG" else ("Below EMA20/EMA50" if weekly_trend else "Mixed"),
        "daily": "Above EMA20/EMA50" if daily_trend and direction == "LONG" else ("Below EMA20/EMA50" if daily_trend else "Mixed"),
        "close": round(daily_close, 2), "ema20": round(ema20_d, 2), "ema50": round(ema50_d, 2),
        "ema20_weekly": round(ema20_w, 2), "ema50_weekly": round(ema50_w, 2),
        "ema20_h4": round(ema20_h4, 2), "ema50_h4": round(ema50_h4, 2),
        "h4_close": round(h4_close, 2), "h4_invalidated": h4_invalidated,
        "pullback": pullback["pullback"], "near_ema20_h4": pullback["near_ema20"],
        "near_ema50_h4": pullback["near_ema50"], "pullback_atr": pullback["pullback_atr"],
        "bounce_confirmed": bounce, "extended": extended,
        "daily_extension_atr": round(float(extension_atr), 2) if math.isfinite(extension_atr) else None,
        "return_60d_pct": round(return_60d * 100, 2),
        "directional_return_60d_pct": round(directional_return * 100, 2),
        "failed_conditions": failed, "trend_alignment": higher_timeframe_trend,
        "h4_reclaimed_ema20": bounce,
        "ema20_distance_h4_pct": round(h4_distance_pct, 3) if h4_distance_pct is not None else None,
    }


def priority(item):
    order = {
        "Confirmed setup": 0,
        "Pullback waiting": 1,
        "Trend OK": 2,
        "Trend invalidated": 3,
        "Data error": 4,
    }
    distance = item.get("ema20_distance_h4_pct")
    distance_key = float(distance) if distance is not None else 999.0
    extension = item.get("daily_extension_atr")
    extension_key = abs(float(extension)) if extension is not None else 999.0
    return (order.get(item.get("status"), 5), -int(item.get("score") or 0), extension_key, distance_key, item.get("ticker", ""))


def main():
    universe = pd.read_csv(TICKERS_FILE)
    if not {"ticker", "sector"}.issubset(universe.columns):
        raise ValueError("tickers.csv musi zawierac kolumny: ticker,sector")

    base_rows = []
    long_rows = []
    short_rows = []
    errors = []

    for _, row in universe.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        sector_value = row["sector"]
        sector = "" if pd.isna(sector_value) else str(sector_value).strip()

        try:
            data = fetch(ticker)
            if data is None:
                reason = "Brak danych dziennych lub 30-minutowych z Yahoo Finance"
                base = {
                    "ticker": ticker,
                    "sector": sector,
                    "data_ok": False,
                    "reason": reason,
                    "long_status": "Data error",
                    "short_status": "Data error",
                    "long_score": None,
                    "short_score": None,
                }
                base_rows.append(base)
                errors.append({"ticker": ticker, "reason": reason})
                continue

            weekly, daily, h4 = data
            if len(weekly) < EMA_SLOW or len(daily) < EMA_SLOW or len(h4) < 65:
                reason = "Za malo danych do EMA20/EMA50 lub H4"
                base = {
                    "ticker": ticker,
                    "sector": sector,
                    "data_ok": False,
                    "reason": reason,
                    "long_status": "Data error",
                    "short_status": "Data error",
                    "long_score": None,
                    "short_score": None,
                }
                base_rows.append(base)
                errors.append({"ticker": ticker, "reason": reason})
                continue

            return_60d = float(daily["Close"].pct_change(60).iloc[-1])
            long_result = evaluate_direction(
                ticker, sector, "LONG", weekly, daily, h4, return_60d
            )
            short_result = evaluate_direction(
                ticker, sector, "SHORT", weekly, daily, h4, return_60d
            )
            long_rows.append(long_result)
            short_rows.append(short_result)

            base_rows.append({
                "ticker": ticker,
                "sector": sector,
                "data_ok": True,
                "close": long_result["close"],
                "ema20": long_result["ema20"],
                "ema50": long_result["ema50"],
                "ema20_weekly": long_result["ema20_weekly"],
                "ema50_weekly": long_result["ema50_weekly"],
                "ema20_h4": long_result["ema20_h4"],
                "ema50_h4": long_result["ema50_h4"],
                "return_60d_pct": long_result["return_60d_pct"],
                "long_status": long_result["status"],
                "short_status": short_result["status"],
                "long_score": long_result["score"],
                "short_score": short_result["score"],
                "long_failed_conditions": long_result["failed_conditions"],
                "short_failed_conditions": short_result["failed_conditions"],
            })
        except Exception as error:
            message = str(error)
            print("Pominieto " + ticker + ": " + message)
            errors.append({"ticker": ticker, "reason": message})
            base_rows.append({
                "ticker": ticker,
                "sector": sector,
                "data_ok": False,
                "reason": message,
                "long_status": "Data error",
                "short_status": "Data error",
                "long_score": None,
                "short_score": None,
            })

    long_candidates = [
        item for item in long_rows
        if item["data_ok"] and item["status"] != "Trend invalidated"
    ]
    short_candidates = [
        item for item in short_rows
        if item["data_ok"] and item["status"] != "Trend invalidated"
    ]
    long_candidates.sort(key=priority)
    short_candidates.sort(key=priority)

    long_setups = [item for item in long_rows if item["qualifies"]]
    short_setups = [item for item in short_rows if item["qualifies"]]
    long_setups.sort(key=priority)
    short_setups.sort(key=priority)

    combined_setups = sorted(long_setups + short_setups, key=priority)
    diagnostics = [
        item for item in base_rows
        if item["ticker"] in DIAGNOSTIC_TICKERS
    ]
    diagnostics.sort(key=lambda item: item["ticker"])
    base_rows.sort(key=lambda item: (
        item.get("data_ok", False),
        max(item.get("long_score") or 0, item.get("short_score") or 0),
    ), reverse=True)

    output = {
        "updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "strategy_version": "Swing Scanner V2 - EMA20/EMA50 LONG/SHORT",
        "universe_count": int(len(universe)),
        "scanned_count": int(sum(1 for item in base_rows if item.get("data_ok"))),
        "data_error_count": int(len(errors)),
        "long_candidate_count": int(len(long_candidates)),
        "short_candidate_count": int(len(short_candidates)),
        "long_confirmed_count": int(len(long_setups)),
        "short_confirmed_count": int(len(short_setups)),
        "qualifying_count": int(len(combined_setups)),
        "long_candidates": long_candidates[:40],
        "short_candidates": short_candidates[:40],
        "long_setups": long_setups[:30],
        "short_setups": short_setups[:30],
        # Compatibility key for any older part of the dashboard.
        "setups": combined_setups[:30],
        "diagnostic_tickers": diagnostics,
        "all_diagnostics": base_rows,
        "data_errors": errors,
    }

    OUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    print("Przeskanowano " + str(output["scanned_count"]) + "/" + str(len(universe)) + " spolek.")
    print("Potwierdzone LONG: " + str(len(long_setups)) + ".")
    print("Potwierdzone SHORT: " + str(len(short_setups)) + ".")
    print("Bledy danych: " + str(len(errors)) + ".")


if __name__ == "__main__":
    main()
