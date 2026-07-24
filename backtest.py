"""
Backtest Modulu - "En iyi sistemi bul"
=======================================
Ayni strateji mantigini GECMIS veride calistirir ve farkli parametre
kombinasyonlarini yaristirip sonuclari karsilastirir.

Nasil calisir:
  - Her sembol icin gecmis mumlar cekilir (limit=1000: 4h'de ~166 gun, 1d'de ~2.7 yil)
  - Zaman icinde ilerlenir; her mumda "o gun sinyal olur muydu?" diye bakilir
    (SADECE o ana kadarki veri kullanilir - gelecege bakma yok)
  - Sinyal olusan yerden itibaren ayni pozisyon modeli simule edilir
    (1/3 TP1 + BE, 1/3 TP2, 1/3 TP3, kotumser ayni-mum kurali)
  - Her parametre kombinasyonu icin: toplam R, islem sayisi, basari, max dusus

Calistirma: GitHub Actions'ta "Backtest" workflow'unu elle tetikle (Run workflow)
veya lokal: python backtest.py

UYARI: Gecmis performans gelecegi garanti etmez. Backtest sonuclari gercek
islemden iyimser sapar (komisyon, kayma, likidite yok sayilir). En iyi cikan
kombinasyonu hemen uygulamak yerine mantikli olup olmadigini dusun - cok dar
bir veri donemine asiri uyum (overfitting) en buyuk tuzaktir.
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

# ---------------- BACKTEST AYARLARI ----------------
BT_INTERVAL = os.environ.get("BT_INTERVAL", "4h")   # test edilecek zaman dilimi
BT_SYMBOL_COUNT = int(os.environ.get("BT_SYMBOL_COUNT", "80"))  # hacme gore ilk N coin
BT_CANDLES = 1000                                    # cekilecek mum sayisi (API max)
WARMUP = 120                                         # ilk sinyale bakmadan once gereken mum
COOLDOWN_BARS = 6                                    # sinyal sonrasi kac mum yeni sinyal aranmaz
MAX_HOLD_BARS = 270                                  # pozisyon en fazla kac mum acik kalir (~45 gun 4h)

# Sabit kurallar (ana sistemle ayni)
MIN_STOP_DIST_PCT = 0.02
MOVE_STOP_TO_BE = True

# Yaristirilacak parametre kombinasyonlari
PARAM_GRID = {
    "ema_period":        [55, 99],
    "rally_min_pct":     [0.40, 0.50, 0.70],
    "touch_tol_pct":     [0.010, 0.015, 0.025],
    "pullback_min_pct":  [0.05, 0.10],
    "min_rr":            [2.0, 3.0],
}
# Toplam kombinasyon: 2*3*3*2*2 = 72

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC_SUMMARY = os.environ.get("TOPIC_SUMMARY")

RALLY_MAX_DAYS = 30


# ---------------- VERI ----------------
def get_top_symbols(n):
    """24s hacme gore ilk n USDT paritesi."""
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
        if base in ("USDC", "FDUSD", "TUSD", "DAI", "EUR", "TRY", "BUSD"):
            continue  # stabil pariteler
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
        if not raw or len(raw) < WARMUP + 50:
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


def bars_per_day(interval):
    return {"1h": 24, "4h": 6, "1d": 1}.get(interval, 6)


# ---------------- STRATEJI (parametrik) ----------------
def check_signal(closes, emas, i, p, interval):
    """i indeksindeki mumda sinyal var mi? Sadece [0..i] verisi kullanilir.
    Return: plan dict veya None"""
    win = int(RALLY_MAX_DAYS * bars_per_day(interval))
    start = max(0, i - win)
    seg = closes[start:i + 1]
    if len(seg) < 5:
        return None

    mi = int(np.argmin(seg))
    if mi >= len(seg) - 2:
        return None
    ma = mi + int(np.argmax(seg[mi:]))
    lo, hi = seg[mi], seg[ma]
    if lo <= 0:
        return None
    if (hi - lo) / lo < p["rally_min_pct"]:
        return None

    entry = closes[i]
    if (hi - entry) / hi < p["pullback_min_pct"]:
        return None

    ema = emas[i]
    if abs(entry - ema) / ema > p["touch_tol_pct"]:
        return None

    # plan
    diff = hi - lo
    tp1 = hi - diff * 0.382
    tp2 = hi
    tp3 = hi + diff * 0.272
    stop = max(hi - diff * 0.786, lo * 0.999)
    if stop >= entry:
        return None
    if (entry - stop) / entry < MIN_STOP_DIST_PCT:
        stop = entry * (1 - MIN_STOP_DIST_PCT)
    risk = entry - stop
    tps = [t for t in (tp1, tp2, tp3) if t > entry]
    if not tps or (min(tps) - entry) / risk < p["min_rr"]:
        return None
    return {"entry": entry, "stop": stop, "risk": risk,
            "tp1": tp1, "tp2": tp2, "tp3": tp3}


def simulate_trade(highs, lows, closes, start_i, plan):
    """start_i'den sonraki mumlarda pozisyonu simule eder. Return: (realized_r, end_i)"""
    entry, risk = plan["entry"], plan["risk"]
    stop = plan["stop"]
    hit = set()
    end = min(len(closes) - 1, start_i + MAX_HOLD_BARS)

    for j in range(start_i + 1, end + 1):
        # kotumser: once stop
        if lows[j] <= stop:
            if not hit:
                return -1.0, j
            r = sum((plan[f"tp{k}"] - entry) / risk for k in hit) / 3.0
            return r, j
        for k in (1, 2, 3):
            if k not in hit and highs[j] >= plan[f"tp{k}"]:
                hit.add(k)
                if k == 1 and MOVE_STOP_TO_BE:
                    stop = entry
        if 3 in hit:
            r = sum((plan[f"tp{k}"] - entry) / risk for k in (1, 2, 3)) / 3.0
            return r, j

    # sure doldu: gerceklesen TP'ler + kalan kismin son fiyattaki degeri
    r = sum((plan[f"tp{k}"] - entry) / risk for k in hit) / 3.0
    open_thirds = 3 - len(hit)
    r += ((closes[end] - entry) / risk) * (open_thirds / 3.0)
    return r, end


def run_config(data, p, interval):
    """Bir parametre kombinasyonunu tum sembollerde calistirir."""
    trades = []
    for symbol, arrs in data.items():
        closes, highs, lows, ema_map = arrs
        emas = ema_map[p["ema_period"]]
        i = WARMUP
        n = len(closes)
        while i < n - 2:
            plan = check_signal(closes, emas, i, p, interval)
            if plan:
                r, end_i = simulate_trade(highs, lows, closes, i, plan)
                trades.append(r)
                i = end_i + COOLDOWN_BARS
            else:
                i += 1

    if not trades:
        return {"trades": 0, "total_r": 0, "avg_r": 0, "win_rate": 0, "max_dd": 0}

    arr = np.array(trades)
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max())
    return {
        "trades": len(arr),
        "total_r": float(arr.sum()),
        "avg_r": float(arr.mean()),
        "win_rate": float((arr > 0).mean() * 100),
        "max_dd": max_dd,
    }


# ---------------- TELEGRAM ----------------
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


# ---------------- ANA ----------------
def main():
    print(f"Backtest: {BT_INTERVAL}, ilk {BT_SYMBOL_COUNT} coin, {BT_CANDLES} mum")
    symbols = get_top_symbols(BT_SYMBOL_COUNT)
    print(f"{len(symbols)} sembol icin veri cekiliyor...")

    ema_periods = sorted(set(PARAM_GRID["ema_period"]))
    data = {}
    for s in symbols:
        df = get_klines(s, BT_INTERVAL)
        if df is None:
            continue
        closes = df["close"].values
        ema_map = {p: df["close"].ewm(span=p, adjust=False).mean().values
                   for p in ema_periods}
        data[s] = (closes, df["high"].values, df["low"].values, ema_map)
        time.sleep(0.03)
    print(f"{len(data)} sembol yuklendi. Kombinasyonlar calisiyor...")

    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*PARAM_GRID.values())]
    results = []
    for idx, p in enumerate(combos):
        res = run_config(data, p, BT_INTERVAL)
        results.append({**p, **res})
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(combos)} tamamlandi")

    # islem sayisi cok dusuk olanlari sirali listeden ele (gurultu)
    valid = [r for r in results if r["trades"] >= 20]
    pool = valid if valid else results
    pool.sort(key=lambda r: r["total_r"], reverse=True)

    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=1)

    # rapor
    period_days = BT_CANDLES / bars_per_day(BT_INTERVAL)
    lines = [f"🧪 <b>BACKTEST SONUCLARI</b>",
             f"Donem: son ~{period_days:.0f} gun | {BT_INTERVAL} | {len(data)} coin | "
             f"{len(combos)} kombinasyon\n",
             "<b>En iyi 5:</b>"]
    for r in pool[:5]:
        lines.append(
            f"EMA{r['ema_period']} | rally%{r['rally_min_pct']*100:.0f} | "
            f"tol%{r['touch_tol_pct']*100:.1f} | pb%{r['pullback_min_pct']*100:.0f} | "
            f"RR{r['min_rr']:.0f}\n"
            f"  → {r['total_r']:+.1f}R | {r['trades']} islem | ort {r['avg_r']:+.2f}R | "
            f"basari %{r['win_rate']:.0f} | max dusus {r['max_dd']:.1f}R")

    lines.append("\n<b>En kotu 3:</b>")
    for r in pool[-3:]:
        lines.append(
            f"EMA{r['ema_period']} | rally%{r['rally_min_pct']*100:.0f} | "
            f"tol%{r['touch_tol_pct']*100:.1f} | pb%{r['pullback_min_pct']*100:.0f} | "
            f"RR{r['min_rr']:.0f} → {r['total_r']:+.1f}R ({r['trades']} islem)")

    # mevcut canli ayarlarin siralamasi
    live = next((r for r in results if r["ema_period"] in (55,) and
                 r["rally_min_pct"] == 0.50 and r["touch_tol_pct"] == 0.015 and
                 r["pullback_min_pct"] == 0.05 and r["min_rr"] == 3.0), None)
    if live:
        rank = pool.index(live) + 1 if live in pool else "-"
        lines.append(f"\n📍 Mevcut canli ayarlar (EMA55): sira {rank}/{len(pool)} | "
                     f"{live['total_r']:+.1f}R | {live['trades']} islem")

    lines.append("\n<i>Uyari: Gecmis performans gelecegi garanti etmez. Tek donemlik "
                 "backteste gore parametre secmek overfitting riskidir - en iyi ilk 5'in "
                 "ORTAK yonlerine bak, tek sampiyona degil.</i>")

    report = "\n".join(lines)
    print(report.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    tg_send(report)
    print("\nTum sonuclar: backtest_results.json")


if __name__ == "__main__":
    main()
