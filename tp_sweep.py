#!/usr/bin/env python3
"""
TP TARAMASI  --  Hangi hedef mesafesi en cok kazandirir?
===========================================================
Ayni girisleri alip TEK TP ile farkli mesafelerde test eder:
  TP = giris + k * risk     (k = 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
Ayrica HIZLI CIKIS: pozisyon en fazla N bar tutulur (6/12/24/48/200).

Amac: kazanma orani ile beklenti arasindaki TAKASI sayilarla gostermek.
Yakin hedef -> yuksek kazanma orani, kucuk kazanc.
Uzak hedef  -> dusuk kazanma orani, buyuk kazanc.

Kurulumlar (CIFT YONLU): ob / fvg / zone
Ayni barda hem stop hem TP -> KAYIP (kotumser). Maliyet dusulur.

UYARI: Yatirim tavsiyesi degildir. Kayma dahil degildir.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://data-api.binance.vision"

RALLY_MIN = 0.30
RALLY_WIN_DAYS = 30
PULLBACK_MIN = 0.05
TOUCH_TOL = 0.02
MIN_STOP_PCT = 0.01
MAX_ZONE_W = 0.08
FEE = 0.0005
FUNDING_8H = 0.0001

R_KATLARI = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TUTMA_LIMITLERI = [6, 12, 24, 48, 200]
KURULUMLAR = ["ob", "fvg", "zone"]


def bars_per_day(iv):
    return {"1h": 24, "2h": 12, "4h": 6, "12h": 2, "1d": 1}[iv]


def top_symbols(n):
    r = requests.get(f"{BASE}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    rows = []
    for t in r.json():
        s = t["symbol"]
        if not s.endswith("USDT"):
            continue
        b = s[:-4]
        if any(b.endswith(x) for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        if "USD" in b and len(b) <= 6:
            continue
        rows.append((s, float(t.get("quoteVolume", 0))))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def klines(symbol, interval, days):
    tf_ms = {"1h": 3600, "2h": 7200, "4h": 14400, "12h": 43200, "1d": 86400}[interval] * 1000
    since = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - days * 86_400_000
    rows = []
    while True:
        r = requests.get(f"{BASE}/api/v3/klines",
                         params={"symbol": symbol, "interval": interval,
                                 "startTime": since, "limit": 1000}, timeout=30)
        if r.status_code != 200:
            break
        b = r.json()
        if not b:
            break
        rows += [x[:6] for x in b]
        since = b[-1][0] + tf_ms
        if len(b) < 1000:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
    for x in ("o", "h", "l", "c", "v"):
        df[x] = df[x].astype(float)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").sort_values("dt").reset_index(drop=True)

def hazirla(df, iv):
    """Sembol basina bir kez: EMAlar, ATR ve her bar icin swing bacagi."""
    c, o, h, l, v = (df[x].values for x in ("c", "o", "h", "l", "v"))
    n = len(c)
    s = pd.Series(c)
    ema55 = s.ewm(span=55, adjust=False).mean().values
    ema99 = s.ewm(span=99, adjust=False).mean().values

    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    atr = np.full(n, np.nan)
    if len(tr) > 14:
        a = tr[:14].mean()
        atr[14] = a
        for i in range(15, n):
            a = (a * 13 + tr[i - 1]) / 14
            atr[i] = a

    win = int(RALLY_WIN_DAYS * bars_per_day(iv))
    okL = np.zeros(n, bool); hiL = np.zeros(n); loL = np.zeros(n)
    hiIL = np.zeros(n, int); loIL = np.zeros(n, int)
    okS = np.zeros(n, bool); hiS = np.zeros(n); loS = np.zeros(n)
    hiIS = np.zeros(n, int); loIS = np.zeros(n, int)

    for i in range(120, n):
        st = max(0, i - win)
        seg = c[st:i + 1]
        mi = int(np.argmin(seg))
        if mi < len(seg) - 2:
            ma = mi + int(np.argmax(seg[mi:]))
            lo_, hi_ = seg[mi], seg[ma]
            if lo_ > 0 and (hi_ - lo_) / lo_ >= RALLY_MIN and (hi_ - c[i]) / hi_ >= PULLBACK_MIN:
                okL[i] = True; hiL[i] = hi_; loL[i] = lo_
                hiIL[i] = st + ma; loIL[i] = st + mi
        ma2 = int(np.argmax(seg))
        if ma2 < len(seg) - 2:
            mi2 = ma2 + int(np.argmin(seg[ma2:]))
            hi_, lo_ = seg[ma2], seg[mi2]
            if lo_ > 0 and (hi_ - lo_) / hi_ >= RALLY_MIN and (c[i] - lo_) / lo_ >= PULLBACK_MIN:
                okS[i] = True; hiS[i] = hi_; loS[i] = lo_
                hiIS[i] = st + ma2; loIS[i] = st + mi2

    return dict(c=c, o=o, h=h, l=l, v=v, ema55=ema55, ema99=ema99, atr=atr,
                okL=okL, hiL=hiL, loL=loL, hiIL=hiIL, loIL=loIL,
                okS=okS, hiS=hiS, loS=loS, hiIS=hiIS, loIS=loIS)


def bolge_bul(d, i, kur, side):
    """Kurulum bolgesi var mi? Return (var_mi, bolge_alt, bolge_ust)."""
    c = d["c"][i]
    if kur == "zone":
        a, b = d["ema55"][i], d["ema99"][i]
        zl, zh = min(a, b), max(a, b)
        return (zl * (1 - TOUCH_TOL) <= c <= zh * (1 + TOUCH_TOL)), zl, zh

    if side == 1:
        hi_i, lo_i = d["hiIL"][i], d["loIL"][i]
    else:
        hi_i, lo_i = d["loIS"][i], d["hiIS"][i]
    if abs(hi_i - lo_i) < 3 or i <= max(hi_i, lo_i):
        return False, 0, 0
    h, l, o, cl, v = d["h"], d["l"], d["o"], d["c"], d["v"]
    a, b = (min(hi_i, lo_i), max(hi_i, lo_i))

    if kur == "fvg":
        for j in range(b, a, -1):
            if j - 2 < 0:
                break
            if side == 1 and l[j] > h[j - 2]:
                zl, zh = h[j - 2], l[j]
            elif side == -1 and h[j] < l[j - 2]:
                zl, zh = h[j], l[j - 2]
            else:
                continue
            if zl <= 0 or (zh - zl) / zl > MAX_ZONE_W or (zh - zl) / zl < 0.005:
                continue
            if zl <= c <= zh:
                return True, zl, zh
        return False, 0, 0

    adaylar = []
    for j in range(b - 1, max(a - 3, 1), -1):
        if side == 1 and cl[j] < o[j] and j + 1 <= b and cl[j + 1] > h[j]:
            zl, zh = l[j], h[j]
        elif side == -1 and cl[j] > o[j] and j + 1 <= b and cl[j + 1] < l[j]:
            zl, zh = l[j], h[j]
        else:
            continue
        if zl <= 0 or (zh - zl) / zl > MAX_ZONE_W:
            continue
        if zl <= c <= zh:
            adaylar.append((j, zl, zh))
    if not adaylar:
        return False, 0, 0
    ref = v[max(0, b - 50):b + 1].mean() or 1.0
    j, zl, zh = max(adaylar, key=lambda x: v[x[0]] / ref)
    return True, zl, zh

def calistir(data, kur, side, k, max_hold, tfh):
    """Tek TP = k*risk, en fazla max_hold bar tut."""
    sonuc = []
    for d in data.values():
        c, h, l = d["c"], d["h"], d["l"]
        n = len(c)
        ok = d["okL"] if side == 1 else d["okS"]
        idx = np.flatnonzero(ok)
        blok = -1
        for i in idx:
            if i <= blok or i >= n - 2:
                continue
            var, zl, zh = bolge_bul(d, i, kur, side)
            if not var:
                continue
            entry = c[i]
            if side == 1:
                hi_, lo_ = d["hiL"][i], d["loL"][i]
                stop = max(hi_ - (hi_ - lo_) * 0.786, zl * 0.998)
                if stop >= entry:
                    stop = zl * 0.998
                if stop >= entry or (entry - stop) / entry < MIN_STOP_PCT:
                    stop = entry * (1 - MIN_STOP_PCT)
                risk = entry - stop
                tp = entry + k * risk
            else:
                hi_, lo_ = d["hiS"][i], d["loS"][i]
                stop = min(lo_ + (hi_ - lo_) * 0.786, zh * 1.002)
                if stop <= entry:
                    stop = zh * 1.002
                if stop <= entry or (stop - entry) / entry < MIN_STOP_PCT:
                    stop = entry * (1 + MIN_STOP_PCT)
                risk = stop - entry
                tp = entry - k * risk
            if risk <= 0:
                continue

            son, bar = None, 0
            for j in range(i + 1, min(i + 1 + max_hold, n)):
                bar = j - i
                stop_vur = (l[j] <= stop) if side == 1 else (h[j] >= stop)
                if stop_vur:
                    son = -1.0
                    break
                tp_vur = (h[j] >= tp) if side == 1 else (l[j] <= tp)
                if tp_vur:
                    son = k
                    break
            if son is None:
                j = min(i + max_hold, n - 1)
                bar = j - i
                son = side * (c[j] - entry) / risk

            sd = risk / entry
            maliyet = 2 * FEE / sd + FUNDING_8H * (bar * tfh / 8) / sd
            sonuc.append(son - maliyet)
            blok = i + bar + 3
    return np.array(sonuc)


def rapor(arr, ad):
    if len(arr) < 10:
        return None
    win = float((arr > 0).mean())
    eq = np.cumsum(arr)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(ad=ad, n=len(arr), win=win, beklenti=float(arr.mean()),
                toplam=float(arr.sum()), max_dusus=dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--days", type=int, default=1460)
    ap.add_argument("--save-json", default="results/tp_sweep.json")
    a = ap.parse_args()

    tfh = {"1h": 1, "2h": 2, "4h": 4, "12h": 12, "1d": 24}[a.tf]
    syms = top_symbols(a.top)
    print(f"{len(syms)} coin, {a.tf}, {a.days} gun indiriliyor...")
    data = {}
    for i, s in enumerate(syms, 1):
        df = klines(s, a.tf, a.days)
        if df is not None and len(df) > 400:
            data[s] = hazirla(df, a.tf)
        if i % 25 == 0:
            print(f"  {i}/{len(syms)}")
        time.sleep(0.03)
    print(f"{len(data)} sembol hazir.")

    tum = []
    print("=" * 96)
    print("  TP MESAFESI TARAMASI  (kazanma orani <-> beklenti takasi)")
    print("=" * 96)
    for kur, side in itertools.product(KURULUMLAR, (1, -1)):
        yon = "LONG" if side == 1 else "SHORT"
        print(f"--- {kur.upper()} {yon} ---")
        print(f"  {'TP':>5} {'tutma':>6} {'islem':>6} {'kazanma':>8} {'beklenti':>9} {'toplam':>9} {'dusus':>8}")
        for k in R_KATLARI:
            for mh in TUTMA_LIMITLERI:
                arr = calistir(data, kur, side, k, mh, tfh)
                r = rapor(arr, f"{kur}_{yon}_TP{k}_hold{mh}")
                if not r:
                    continue
                r.update(kurulum=kur, yon=yon, tp_r=k, tutma=mh)
                tum.append(r)
                print(f"  {k:>5.1f} {mh:>6} {r['n']:>6} {r['win']*100:>7.1f}% "
                      f"{r['beklenti']:>+9.3f} {r['toplam']:>+9.1f} {r['max_dusus']:>8.1f}")

    if tum:
        tum.sort(key=lambda x: -x["beklenti"])
        print("=" * 96)
        print("  EN IYI 15 (islem basi beklentiye gore)")
        print("=" * 96)
        for r in tum[:15]:
            print(f"  {r['ad']:<30} {r['n']:>5} islem | kazanma %{r['win']*100:>5.1f} | "
                  f"beklenti {r['beklenti']:+.3f}R | toplam {r['toplam']:+.1f}R")
        yeterli = [r for r in tum if r["n"] >= 100]
        if yeterli:
            print("  100+ ISLEMLI EN IYI 5 (guvenilir orneklem):")
            for r in yeterli[:5]:
                se = 1.96 * np.sqrt(r["win"] * (1 - r["win"]) / r["n"])
                print(f"  {r['ad']:<30} kazanma %{r['win']*100:.1f} (+-{se*100:.1f}) | "
                      f"beklenti {r['beklenti']:+.3f}R | {r['n']} islem")
        print("  KAZANMA ORANI vs BEKLENTI (tum kurulumlarin ortalamasi):")
        for k in R_KATLARI:
            g = [r for r in tum if r["tp_r"] == k]
            if g:
                print(f"    TP {k:>3.1f}R -> ort kazanma %{np.mean([x['win'] for x in g])*100:>5.1f} | "
                      f"ort beklenti {np.mean([x['beklenti'] for x in g]):+.3f}R")
        print("  -> Onemli olan KAZANMA ORANI degil, BEKLENTIDIR.")

    os.makedirs(os.path.dirname(a.save_json) or ".", exist_ok=True)
    json.dump({"meta": {"coin": len(data), "tf": a.tf, "days": a.days,
                        "uretim": str(pd.Timestamp.now(tz="UTC"))},
               "sonuclar": tum}, open(a.save_json, "w"), indent=1, default=str)
    print(f"Kaydedildi: {a.save_json}")
    print("UYARI: Kayma dahil degildir; gecmis performans gelecegi garanti etmez.")


if __name__ == "__main__":
    main()
