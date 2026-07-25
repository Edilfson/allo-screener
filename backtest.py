"""
Backtest v2 - Genisletilmis parametre yarisi
=============================================
Yaristirilan boyutlar:
  - MA turu: EMA21 / EMA55 / EMA99 / EMA144 / EMA200 / SMA200 / ZONE(EMA55-99 bandi)
  - Temas toleransi: %1.5 / %2.5
  - Min RR: 2 / 3
  - Giris modu: temasta (sinyal mumu kapanisi) vs onayda (sonraki 3 mumdaki ilk yesil kapanis)
  - Stop modu: fib (0.786/swing low) vs atr (giris - 1.5*ATR14)
  - Kalite filtresi: acik/kapali (uzun alt fitil VEYA MA ustunde kapanis sarti)
Sabit: rally >= %50 (iki onceki testin ortak bulgusu), pullback >= %5.

Ek: her islem, acildigi andaki BTC rejimiyle (1D EMA99 ustu=boga / alti=ayi)
etiketlenir; rapor boga/ayi kirilimini gosterir. Rejim sinyal KAPATMAZ, sadece olcum.

Calistirma: Actions > Backtest > Run workflow (interval: 4h veya 1d)

UYARI: Gecmis performans gelecegi garanti etmez; komisyon/kayma dahil degildir.
En iyi tek kombinasyona degil, iyi kumelerin ORTAK ozelliklerine bak.
"""

import itertools
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision"

BT_INTERVAL = os.environ.get("BT_INTERVAL", "4h")
BT_SYMBOL_COUNT = int(os.environ.get("BT_SYMBOL_COUNT", "60"))
BT_CANDLES = 1000
WARMUP = 210
COOLDOWN_BARS = 6
MAX_HOLD_BARS = 270

RALLY_MIN_PCT = 0.50
PULLBACK_MIN_PCT = 0.05
RALLY_MAX_DAYS = 30
MIN_STOP_DIST_PCT = 0.02
MOVE_STOP_TO_BE = True

MA_OPTIONS = ["ema21", "ema55", "ema99", "ema144", "ema200", "sma200", "zone5599"]
PARAM_GRID = {
    "ma": MA_OPTIONS,
    "touch_tol": [0.015, 0.025],
    "min_rr": [2.0, 3.0],
    "entry_mode": ["temasta", "onayda"],
    "stop_mode": ["fib", "atr"],
    "quality": [False, True],
}

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC_SUMMARY = os.environ.get("TOPIC_SUMMARY")


def get_top_symbols(n):
    r = requests.get(f"{BASE_URL}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    rows = []
    for t in r.json():
        s = t["symbol"]
        if not s.endswith("USDT"):
            continue
        base = s[:-4]
        if any(base.endswith(x) for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        if base in ("USDC", "FDUSD", "TUSD", "DAI", "EUR", "TRY", "BUSD", "USDP"):
            continue
        rows.append((s, float(t.get("quoteVolume", 0))))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def get_klines(symbol, interval, limit=BT_CANDLES):
    try:
        r = requests.get(f"{BASE_URL}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=30)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not raw or len(raw) < WARMUP + 60:
            return None
        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df
    except Exception:
        return None


def bars_per_day(interval):
    return {"1h": 24, "4h": 6, "1d": 1}.get(interval, 6)


def calc_atr(highs, lows, closes, period=14):
    tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
    atr = np.full(len(closes), np.nan)
    if len(tr) >= period:
        a = tr[:period].mean()
        atr[period] = a
        for i in range(period + 1, len(closes)):
            a = (a * (period - 1) + tr[i - 1]) / period
            atr[i] = a
    return atr


def precompute_symbol(df, interval):
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["close_time"].values.astype("datetime64[s]").astype(np.int64)
    mas = {}
    s = pd.Series(closes)
    for p in (21, 55, 99, 144, 200):
        mas[f"ema{p}"] = s.ewm(span=p, adjust=False).mean().values
    mas["sma200"] = s.rolling(200).mean().values
    atr = calc_atr(highs, lows, closes)
    win = int(RALLY_MAX_DAYS * bars_per_day(interval))
    n = len(closes)
    rally_ok = np.zeros(n, dtype=bool)
    sw_hi = np.zeros(n)
    sw_lo = np.zeros(n)
    for i in range(WARMUP, n):
        seg = closes[max(0, i - win):i + 1]
        mi = int(np.argmin(seg))
        if mi >= len(seg) - 2:
            continue
        ma_i = mi + int(np.argmax(seg[mi:]))
        lo, hi = seg[mi], seg[ma_i]
        if lo <= 0 or (hi - lo) / lo < RALLY_MIN_PCT:
            continue
        if (hi - closes[i]) / hi < PULLBACK_MIN_PCT:
            continue
        rally_ok[i] = True
        sw_hi[i] = hi
        sw_lo[i] = lo
    return {"closes": closes, "opens": opens, "highs": highs, "lows": lows, "times": times, "mas": mas, "atr": atr, "rally_ok": rally_ok, "sw_hi": sw_hi, "sw_lo": sw_lo}


def get_btc_regime_lookup(interval):
    df = get_klines("BTCUSDT", "1d")
    if df is None:
        return lambda ts: None
    closes = df["close"].values
    ema99 = pd.Series(closes).ewm(span=99, adjust=False).mean().values
    times = df["close_time"].values.astype("datetime64[s]").astype(np.int64)
    above = closes >= ema99
    def lookup(ts):
        idx = np.searchsorted(times, ts, side="right") - 1
        if idx < 0:
            return None
        return bool(above[idx])
    return lookup


def ma_touch(sym, i, p):
    close = sym["closes"][i]
    tol = p["touch_tol"]
    if p["ma"] == "zone5599":
        a, b = sym["mas"]["ema55"][i], sym["mas"]["ema99"][i]
        lo_b, hi_b = min(a, b), max(a, b)
        return lo_b * (1 - tol) <= close <= hi_b * (1 + tol)
    ma = sym["mas"][p["ma"]][i]
    if np.isnan(ma):
        return False
    return abs(close - ma) / ma <= tol


def quality_ok(sym, i, p):
    if not p["quality"]:
        return True
    o, c = sym["opens"][i], sym["closes"][i]
    lo = sym["lows"][i]
    body = abs(c - o)
    lower_wick = min(o, c) - lo
    if body > 0 and lower_wick >= body:
        return True
    if p["ma"] == "zone5599":
        ref = max(sym["mas"]["ema55"][i], sym["mas"]["ema99"][i])
    else:
        ref = sym["mas"][p["ma"]][i]
    return (not np.isnan(ref)) and c >= ref


def build_plan(sym, i, entry_i, p):
    hi, lo = sym["sw_hi"][i], sym["sw_lo"][i]
    entry = sym["closes"][entry_i]
    diff = hi - lo
    if diff <= 0 or entry <= 0:
        return None
    tp1 = hi - diff * 0.382
    tp2 = hi
    tp3 = hi + diff * 0.272
    if p["stop_mode"] == "atr":
        atr = sym["atr"][i]
        if np.isnan(atr):
            return None
        stop = entry - 1.5 * atr
    else:
        stop = max(hi - diff * 0.786, lo * 0.999)
    if stop >= entry:
        return None
    if (entry - stop) / entry < MIN_STOP_DIST_PCT:
        stop = entry * (1 - MIN_STOP_DIST_PCT)
    risk = entry - stop
    tps = [t for t in (tp1, tp2, tp3) if t > entry]
    if not tps or (min(tps) - entry) / risk < p["min_rr"]:
        return None
    return {"entry": entry, "stop": stop, "risk": risk, "tp1": tp1, "tp2": tp2, "tp3": tp3}


def find_entry_bar(sym, i, p):
    if p["entry_mode"] == "temasta":
        return i
    n = len(sym["closes"])
    for j in range(i + 1, min(i + 4, n)):
        if sym["closes"][j] > sym["opens"][j]:
            return j
    return None


def simulate_trade(sym, start_i, plan):
    highs, lows, closes = sym["highs"], sym["lows"], sym["closes"]
    entry, risk = plan["entry"], plan["risk"]
    stop = plan["stop"]
    hit = set()
    end = min(len(closes) - 1, start_i + MAX_HOLD_BARS)
    for j in range(start_i + 1, end + 1):
        if lows[j] <= stop:
            if not hit:
                return -1.0, j
            return sum((plan[f"tp{k}"] - entry) / risk for k in hit) / 3.0, j
        for k in (1, 2, 3):
            if k not in hit and highs[j] >= plan[f"tp{k}"]:
                hit.add(k)
                if k == 1 and MOVE_STOP_TO_BE:
                    stop = entry
        if 3 in hit:
            return sum((plan[f"tp{k}"] - entry) / risk for k in (1, 2, 3)) / 3.0, j
    r = sum((plan[f"tp{k}"] - entry) / risk for k in hit) / 3.0
    r += ((closes[end] - entry) / risk) * ((3 - len(hit)) / 3.0)
    return r, end


def run_config(data, p, btc_lookup):
    trades = []
    for sym in data.values():
        idxs = np.where(sym["rally_ok"])[0]
        n = len(sym["closes"])
        blocked_until = -1
        for i in idxs:
            if i <= blocked_until or i >= n - 2:
                continue
            if not ma_touch(sym, i, p) or not quality_ok(sym, i, p):
                continue
            entry_i = find_entry_bar(sym, i, p)
            if entry_i is None or entry_i >= n - 1:
                continue
            plan = build_plan(sym, i, entry_i, p)
            if not plan:
                continue
            r, end_i = simulate_trade(sym, entry_i, plan)
            regime = btc_lookup(int(sym["times"][i]))
            trades.append((r, regime))
            blocked_until = end_i + COOLDOWN_BARS
    if not trades:
        return {"trades": 0, "total_r": 0.0, "avg_r": 0.0, "win_rate": 0.0, "max_dd": 0.0, "bull_r": 0.0, "bull_n": 0, "bear_r": 0.0, "bear_n": 0}
    arr = np.array([t[0] for t in trades])
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    bull = [r for r, reg in trades if reg is True]
    bear = [r for r, reg in trades if reg is False]
    return {"trades": len(arr), "total_r": float(arr.sum()), "avg_r": float(arr.mean()), "win_rate": float((arr > 0).mean() * 100), "max_dd": float((peak - equity).max()), "bull_r": float(sum(bull)), "bull_n": len(bull), "bear_r": float(sum(bear)), "bear_n": len(bear)}


def tg_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:4090], "parse_mode": "HTML"}
    if TOPIC_SUMMARY:
        data["message_thread_id"] = TOPIC_SUMMARY
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data=data, timeout=20)
    except Exception as e:
        print("TG hata:", e)


def cfg_label(r):
    q = "kalite+" if r["quality"] else "kalite-"
    return f"{r['ma']} tol%{r['touch_tol']*100:.1f} RR{r['min_rr']:.0f} {r['entry_mode']} {r['stop_mode']} {q}"


def main():
    print(f"Backtest v2: {BT_INTERVAL}, ilk {BT_SYMBOL_COUNT} coin")
    symbols = get_top_symbols(BT_SYMBOL_COUNT)
    btc_lookup = get_btc_regime_lookup(BT_INTERVAL)
    data = {}
    for s in symbols:
        df = get_klines(s, BT_INTERVAL)
        if df is None:
            continue
        data[s] = precompute_symbol(df, BT_INTERVAL)
        time.sleep(0.03)
    print(f"{len(data)} sembol hazir. Kombinasyonlar calisiyor...")
    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    results = []
    for idx, p in enumerate(combos):
        results.append({**p, **run_config(data, p, btc_lookup)})
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(combos)}")
    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=1)
    valid = [r for r in results if r["trades"] >= 15]
    pool = sorted(valid if valid else results, key=lambda r: -r["total_r"])
    period_days = BT_CANDLES / bars_per_day(BT_INTERVAL)
    L = [f"BACKTEST v2 | ~{period_days:.0f} gun | {BT_INTERVAL} | {len(data)} coin | {len(combos)} komb.\n", "<b>En iyi 8:</b>"]
    for r in pool[:8]:
        L.append(f"{cfg_label(r)}\n  -> {r['total_r']:+.1f}R | {r['trades']}isl | ort {r['avg_r']:+.2f}R | win%{r['win_rate']:.0f} | dd{r['max_dd']:.1f}\n  boga {r['bull_r']:+.1f}R ({r['bull_n']})  ayi {r['bear_r']:+.1f}R ({r['bear_n']})")
    L.append("\n<b>Boyut etkileri (ort. toplam R):</b>")
    for param in keys:
        vals = []
        for v in PARAM_GRID[param]:
            g = [r for r in results if r[param] == v]
            vals.append(f"{v}:{np.mean([x['total_r'] for x in g]):+.1f}")
        L.append(f"  {param}: " + " | ".join(vals))
    tb_r = sum(r["bull_r"] for r in results)
    tb_n = sum(r["bull_n"] for r in results)
    ta_r = sum(r["bear_r"] for r in results)
    ta_n = sum(r["bear_n"] for r in results)
    if tb_n or ta_n:
        L.append(f"\n<b>Rejim kirilimi:</b> boga: {tb_n} islem, islem basi {tb_r/max(tb_n,1):+.2f}R | ayi: {ta_n} islem, islem basi {ta_r/max(ta_n,1):+.2f}R")
    L.append("\n<i>Uyari: tek doneme asiri uyum riski - kumelerin ortak yonune bak, iki zaman diliminde de tutarli olani sec.</i>")
    report = "\n".join(L)
    print(report.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    tg_send(report)


if __name__ == "__main__":
    main()
