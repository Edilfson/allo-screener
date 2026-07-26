"""
Backtest v4 - Strateji + HTF + bolge kalitesi + trailing yarisi
====================================================
Zaman dilimleri (ayri ayri calistir): 1h, 2h, 4h, 12h, 1d
Giris stratejileri (7):
  - ema7 / ema21 / ema55 / ema99 pullback temasi
  - zone5599 (EMA55-99 bandi)
  - fvg: rally bacagindaki dolmamis yukselis FVG'sine (fair value gap) geri test
  - ob: order block - guclu yukselisten onceki son ayi mumunun bolgesine geri test
TP stilleri (4):
  - klasik: fib382 / onceki zirve / 1.272 uzatma, 1/3-1/3-1/3
  - r_katlari: +2R / +4R / +6R, 1/3-1/3-1/3
  - kosucu: fib382 %50 / zirve %25 / 1.618 uzatma %25
  - tek_hedef: tamami onceki zirvede
Stop stilleri (3):
  - fib786: zirve - 0.786*bacak (swing low taban)
  - atr2: giris - 2*ATR14
  - yapi: MA stratejilerinde fib786/swing-low; FVG/OB'de bolgenin alti
Sabit: rally >= %50, pullback >= %5, min RR 2 (en yakin TP), tolerans %2.
Toplam: 7 x 4 x 3 = 84 kombinasyon / zaman dilimi.
Sonuc: backtest_results.json + repoya results/bt3_<interval>.json commit edilir.

UYARI: Gecmis performans gelecegi garanti etmez; komisyon/kayma dahil degildir.
"""

import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision"
BT_INTERVAL = os.environ.get("BT_INTERVAL", "4h")
BT_SYMBOL_COUNT = int(os.environ.get("BT_SYMBOL_COUNT", "60"))
BT_CANDLES = 1000
WARMUP = 120
COOLDOWN_BARS = 6
MAX_HOLD_BARS = 270

RALLY_MIN_PCT = 0.50
PULLBACK_MIN_PCT = 0.05
RALLY_MAX_DAYS = 30
TOUCH_TOL = 0.02
MIN_RR = 2.0
MIN_STOP_DIST_PCT = 0.02

STRATEGIES = ["ema21", "ema55", "ema99", "zone5599", "fvg", "ob"]
TP_STYLES = ["klasik", "kosucu"]          # r_katlari/tek_hedef elendi
STOP_STYLES = ["fib786"]                  # ATR elendi
HTF_MODES = ["yok", "1d_dususte_alma"]    # coinin KENDI gunluk trend filtresi
ZONE_QUALITY = [False, True]              # OB/FVG bolge kalitesi
TRAIL_MODES = ["yok", "ema21"]            # ilk TP sonrasi iz suren stop
# 6 x 2 x 1 x 2 x 2 x 2 = 96 kombinasyon

MAX_ZONE_WIDTH_PCT = 0.08
MIN_FVG_GAP_PCT = 0.005
TRAIL_BUFFER = 0.01

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC_SUMMARY = os.environ.get("TOPIC_SUMMARY")


def bars_per_day(iv):
    return {"1h": 24, "2h": 12, "4h": 6, "12h": 2, "1d": 1}.get(iv, 6)


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
        r = requests.get(f"{BASE_URL}/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=30)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not raw or len(raw) < WARMUP + 60:
            return None
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df
    except Exception:
        return None


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


def precompute(df, iv):
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    mas = {}
    s = pd.Series(closes)
    for p in (21, 55, 99):
        mas[f"ema{p}"] = s.ewm(span=p, adjust=False).mean().values
    atr = calc_atr(highs, lows, closes)
    win = int(RALLY_MAX_DAYS * bars_per_day(iv))
    n = len(closes)
    rally_ok = np.zeros(n, dtype=bool)
    sw_hi = np.zeros(n)
    sw_lo = np.zeros(n)
    hi_idx = np.zeros(n, dtype=int)
    lo_idx = np.zeros(n, dtype=int)
    for i in range(WARMUP, n):
        st = max(0, i - win)
        seg = closes[st:i + 1]
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
        hi_idx[i] = st + ma_i
        lo_idx[i] = st + mi
    times = df["close_time"].values.astype("datetime64[s]").astype(np.int64) \
        if "close_time" in df.columns else np.arange(n)
    return {"c": closes, "o": opens, "h": highs, "l": lows, "v": df["volume"].values,
            "mas": mas, "atr": atr, "times": times,
            "rally_ok": rally_ok, "sw_hi": sw_hi, "sw_lo": sw_lo,
            "hi_idx": hi_idx, "lo_idx": lo_idx}


def daily_trend_map(symbol):
    """Coinin KENDI 1D trendi haritasi."""
    d = get_klines(symbol, "1d", 400)
    if d is None:
        return None
    c = d["close"].values
    e55 = pd.Series(c).ewm(span=55, adjust=False).mean().values
    e99 = pd.Series(c).ewm(span=99, adjust=False).mean().values
    t = d["close_time"].values.astype("datetime64[s]").astype(np.int64)
    return {"t": t, "dn": (c < e55) & (c < e99)}


def htf_ok(dmap, ts, mode):
    if mode == "yok" or dmap is None:
        return True
    idx = np.searchsorted(dmap["t"], ts, side="right") - 1
    if idx < 0:
        return True
    return not bool(dmap["dn"][idx])      # gunluk dususteyse alma


def entry_check(sym, i, strat, zq=False):
    """Return: (ok, zone_lo) - zone_lo 'yapi' stop icin yapisal taban."""
    c = sym["c"][i]
    if strat == "zone5599":
        a, b = sym["mas"]["ema55"][i], sym["mas"]["ema99"][i]
        zl, zh = min(a, b), max(a, b)
        return (zl * (1 - TOUCH_TOL) <= c <= zh * (1 + TOUCH_TOL)), zl
    if strat.startswith("ema"):
        m = sym["mas"][strat][i]
        if np.isnan(m):
            return False, 0.0
        return (abs(c - m) / m <= TOUCH_TOL), m
    hi_i, lo_i = sym["hi_idx"][i], sym["lo_idx"][i]
    if hi_i <= lo_i + 2 or i <= hi_i:
        return False, 0.0
    h, l = sym["h"], sym["l"]
    if strat == "fvg":
        for j in range(hi_i, lo_i + 1, -1):
            if j - 2 < 0:
                break
            if l[j] > h[j - 2]:
                zl, zh = h[j - 2], l[j]
                if zq and zl > 0:
                    gap = (zh - zl) / zl
                    if gap < MIN_FVG_GAP_PCT or gap > MAX_ZONE_WIDTH_PCT:
                        continue
                if i > hi_i + 1 and l[hi_i + 1:i].min() < zl:
                    continue
                if zl <= c <= zh:
                    return True, zl
        return False, 0.0
    if strat == "ob":
        o, cl, v = sym["o"], sym["c"], sym["v"]
        adaylar = []
        for j in range(hi_i - 1, max(lo_i - 3, 1), -1):
            if cl[j] < o[j] and j + 1 <= hi_i and cl[j + 1] > h[j]:
                zl, zh = l[j], h[j]
                if zq and (zl <= 0 or (zh - zl) / zl > MAX_ZONE_WIDTH_PCT):
                    continue
                if i > hi_i + 1 and l[hi_i + 1:i].min() < zl:
                    continue
                if zl <= c <= zh:
                    adaylar.append((j, zl))
        if not adaylar:
            return False, 0.0
        if zq:
            ref = v[max(0, hi_i - 50):hi_i + 1].mean() or 1.0
            j, zl = max(adaylar, key=lambda a: v[a[0]] / ref)
            return True, zl
        return True, adaylar[0][1]
    return False, 0.0


def build_plan(sym, i, strat, tp_style, stop_style, zone_lo):
    hi, lo = sym["sw_hi"][i], sym["sw_lo"][i]
    entry = sym["c"][i]
    diff = hi - lo
    if diff <= 0 or entry <= 0:
        return None

    if stop_style == "fib786":
        stop = max(hi - diff * 0.786, lo * 0.999)
    elif stop_style == "atr2":
        a = sym["atr"][i]
        if np.isnan(a):
            return None
        stop = entry - 2.0 * a
    else:
        if strat in ("fvg", "ob"):
            stop = zone_lo * 0.998
        else:
            stop = max(hi - diff * 0.786, lo * 0.999)
    if stop >= entry:
        return None
    if (entry - stop) / entry < MIN_STOP_DIST_PCT:
        stop = entry * (1 - MIN_STOP_DIST_PCT)
    risk = entry - stop

    fib382 = hi - diff * 0.382
    ext1272 = hi + diff * 0.272
    ext1618 = hi + diff * 0.618
    if tp_style == "klasik":
        raw = [(fib382, 1 / 3), (hi, 1 / 3), (ext1272, 1 / 3)]
    elif tp_style == "r_katlari":
        raw = [(entry + 2 * risk, 1 / 3), (entry + 4 * risk, 1 / 3), (entry + 6 * risk, 1 / 3)]
    elif tp_style == "kosucu":
        raw = [(fib382, 0.5), (hi, 0.25), (ext1618, 0.25)]
    else:
        raw = [(hi, 1.0)]

    tps = [(p, w) for p, w in raw if p > entry]
    if not tps:
        return None
    tw = sum(w for _, w in tps)
    tps = sorted([(p, w / tw) for p, w in tps])
    if (tps[0][0] - entry) / risk < MIN_RR:
        return None
    return {"entry": entry, "stop": stop, "risk": risk, "tps": tps}


def simulate(sym, start_i, plan, trail="yok"):
    h, l, c = sym["h"], sym["l"], sym["c"]
    entry, risk = plan["entry"], plan["risk"]
    stop = plan["stop"]
    tps = plan["tps"]
    hit = [False] * len(tps)
    realized = 0.0
    end = min(len(c) - 1, start_i + MAX_HOLD_BARS)
    for j in range(start_i + 1, end + 1):
        if l[j] <= stop:
            rem = sum(w for k, (_, w) in enumerate(tps) if not hit[k])
            if not any(hit):
                return realized - rem * 1.0, j
            return realized + rem * (stop - entry) / risk, j
        for k, (tp, w) in enumerate(tps):
            if not hit[k] and h[j] >= tp:
                hit[k] = True
                realized += w * (tp - entry) / risk
                stop = max(stop, entry)
        if trail == "ema21" and any(hit):
            e = sym["mas"]["ema21"][j]
            if not np.isnan(e):
                stop = max(stop, e * (1 - TRAIL_BUFFER))
        if all(hit):
            return realized, j
    rem = sum(w for k, (_, w) in enumerate(tps) if not hit[k])
    realized += rem * (c[end] - entry) / risk
    return realized, end


def run_config(data, dmaps, strat, tp_style, stop_style, htf, zq, trail):
    trades = []
    for name, sym in data.items():
        idxs = np.where(sym["rally_ok"])[0]
        n = len(sym["c"])
        blocked = -1
        for i in idxs:
            if i <= blocked or i >= n - 2:
                continue
            if not htf_ok(dmaps.get(name), int(sym["times"][i]), htf):
                continue
            ok, zone_lo = entry_check(sym, i, strat, zq)
            if not ok:
                continue
            plan = build_plan(sym, i, strat, tp_style, stop_style, zone_lo)
            if not plan:
                continue
            r, end_i = simulate(sym, i, plan, trail)
            trades.append(r)
            blocked = end_i + COOLDOWN_BARS
    if not trades:
        return {"trades": 0, "total_r": 0.0, "avg_r": 0.0, "win_rate": 0.0, "max_dd": 0.0}
    arr = np.array(trades)
    eq = np.cumsum(arr)
    pk = np.maximum.accumulate(eq)
    return {"trades": len(arr), "total_r": float(arr.sum()), "avg_r": float(arr.mean()),
            "win_rate": float((arr > 0).mean() * 100), "max_dd": float((pk - eq).max())}


def tg_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:4090], "parse_mode": "HTML"}
    if TOPIC_SUMMARY:
        data["message_thread_id"] = TOPIC_SUMMARY
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data=data, timeout=20)
    except Exception as e:
        print("TG hata:", e)


def main():
    print(f"Backtest v3: {BT_INTERVAL}, ilk {BT_SYMBOL_COUNT} coin")
    symbols = get_top_symbols(BT_SYMBOL_COUNT)
    data = {}
    for s in symbols:
        df = get_klines(s, BT_INTERVAL)
        if df is None:
            continue
        data[s] = precompute(df, BT_INTERVAL)
        time.sleep(0.03)
    print(f"{len(data)} sembol hazir.")

    dmaps = {}
    print("Gunluk trend haritalari cekiliyor...")
    for s in data:
        dmaps[s] = daily_trend_map(s)
        time.sleep(0.03)

    results = []
    combos = list(itertools.product(STRATEGIES, TP_STYLES, STOP_STYLES,
                                    HTF_MODES, ZONE_QUALITY, TRAIL_MODES))
    for idx, (st, tp, sp, htf, zq, tr) in enumerate(combos):
        res = run_config(data, dmaps, st, tp, sp, htf, zq, tr)
        results.append({"interval": BT_INTERVAL, "strategy": st, "tp_style": tp,
                        "stop_style": sp, "htf": htf, "zone_kalite": zq,
                        "trail": tr, **res})
        if (idx + 1) % 12 == 0:
            print(f"  {idx+1}/{len(combos)}")

    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=1)

    valid = [r for r in results if r["trades"] >= 15]
    pool = sorted(valid if valid else results, key=lambda r: -r["total_r"])
    days = BT_CANDLES / bars_per_day(BT_INTERVAL)
    L = [f"BACKTEST v4 | {BT_INTERVAL} (~{days:.0f} gun) | {len(data)} coin | {len(combos)} komb.\n",
         "<b>En iyi 8:</b>"]
    for r in pool[:8]:
        L.append(f"{r['strategy']} {r['tp_style']} htf:{r['htf']} "
                 f"kalite:{'+' if r['zone_kalite'] else '-'} trail:{r['trail']}\n"
                 f"  -> {r['total_r']:+.1f}R | {r['trades']}isl | ort {r['avg_r']:+.2f}R | "
                 f"win%{r['win_rate']:.0f} | dd{r['max_dd']:.1f}")
    L.append("\n<b>Boyut etkileri (ort toplam R):</b>")
    for param, opts in [("strategy", STRATEGIES), ("tp_style", TP_STYLES),
                        ("htf", HTF_MODES), ("zone_kalite", ZONE_QUALITY),
                        ("trail", TRAIL_MODES)]:
        vals = []
        for v in opts:
            g = [r for r in results if r[param] == v]
            vals.append(f"{v}:{np.mean([x['total_r'] for x in g]):+.1f}")
        L.append(f"  {param}: " + " | ".join(vals))
    L.append("\n<i>Komisyon/kayma dahil degil; tek doneme asiri uyuma dikkat.</i>")
    report = "\n".join(L)
    print(report.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    tg_send(report)


if __name__ == "__main__":
    main()
