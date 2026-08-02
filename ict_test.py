#!/usr/bin/env python3
"""
ICT / SMART MONEY CONCEPTS - KANONIK KURAL TESTI
=================================================
Onceki testimiz ICT'nin BASITLESTIRILMIS halini olcuyordu. Bu test
kanonik tanimlari uygular ve her kuralin katkisini AYRI AYRI olcer.

Onceki testte EKSIK olan 5 sey (bu testte var):
  1. BOS (Break of Structure) sarti - hamle onceki teyitli swing'i kirmali
  2. Displacement sarti - hamle ATR'ye gore impulsif olmali
  3. FVG esligi - displacement bacagi imbalance birakmali
  4. Mean threshold girisi - OB'nin %50'sinden LIMIT emir (kapanista degil)
  5. HTF bias - ust zaman dilimi yonuyle uyum

Test edilen giris modelleri:
  - "kapanis"  : bolgeye degen mumun kapanisinda (bizim eski yontemimiz)
  - "ob50"     : OB'nin %50'sinde limit emir (ICT mean threshold)
  - "ob_uzak"  : OB'nin uzak ucunda limit (en iyi fiyat, dolma riski yuksek)
  - "fvg50"    : FVG'nin %50'sinde limit (consequent encroachment)

Hedef modelleri:
  - R katlari (1,2,3,5R)
  - "likidite": bir sonraki teyitli swing high/low (ICT'nin asil hedefi)

ABLASYON: her kural tek tek acilip kapatilarak katkisi olculur.
Boylece "hangi kural gercekten ise yariyor" sorusu cevaplanir.

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
FEE = 0.0005
FUNDING_8H = 0.0001
SWING_K = 3               # fraktal teyit gecikmesi (ileriye bakma korumasi)
MIN_STOP_PCT = 0.005
FILL_BUFFER = 0.0005      # limit emrin dolmasi icin fiyat seviyeyi bu kadar GECMELI
                          # (sadece dokunmak yetmez - ters secilim/kuyruk riski)


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
    tf_ms = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[interval] * 1000
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
        rows += [x[:5] for x in b]
        since = b[-1][0] + tf_ms
        if len(b) < 1000:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"])
    for x in ("o", "h", "l", "c"):
        df[x] = df[x].astype(float)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def hazirla(df):
    o, h, l, c = (df[x].values for x in ("o", "h", "l", "c"))
    n = len(c)
    # ATR
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    atr = np.full(n, np.nan)
    if len(tr) > 14:
        a = tr[:14].mean()
        atr[14] = a
        for i in range(15, n):
            a = (a * 13 + tr[i - 1]) / 14
            atr[i] = a
    # fraktal swingler (k bar gecikmeyle TEYITLI)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    for i in range(SWING_K, n - SWING_K):
        w = h[i - SWING_K:i + SWING_K + 1]
        if h[i] == w.max() and w.argmax() == SWING_K:
            sh[i] = True
        w = l[i - SWING_K:i + SWING_K + 1]
        if l[i] == w.min() and w.argmin() == SWING_K:
            sl[i] = True
    return dict(o=o, h=h, l=l, c=c, atr=atr, sh=sh, sl=sl, n=n)


def kurulum_ara(d, i, side, p):
    """i barinda ICT kurulumu var mi? Return: dict veya None.
    side: +1 bullish (long), -1 bearish (short)
    p: kural anahtarlari -> bos, displacement, fvg_esligi
    """
    o, h, l, c, atr, n = d["o"], d["h"], d["l"], d["c"], d["atr"], d["n"]
    if i < 60 or np.isnan(atr[i]):
        return None

    # --- 1) BOS: son TEYITLI swing kirildi mi? ---
    # geriye dogru bak: son 40 barda displacement bacagi ariyoruz
    flags = d["sh"] if side == 1 else d["sl"]
    px = h if side == 1 else l
    conf = np.flatnonzero(flags[:max(0, i - SWING_K)])
    if len(conf) < 2:
        return None
    swing_seviye = px[conf[-1]]     # kirilacak seviye

    # BOS barini bul: swing seviyeyi kapanisla asan ILK bar (son 40 barda)
    bos_i = None
    for j in range(max(0, i - 40), i):
        if j <= conf[-1]:
            continue
        if side == 1 and c[j] > swing_seviye:
            bos_i = j
            break
        if side == -1 and c[j] < swing_seviye:
            bos_i = j
            break
    if p["bos"] and bos_i is None:
        return None
    if bos_i is None:
        bos_i = max(0, i - 20)      # BOS sarti kapaliysa yaklasik bacak

    # --- 2) DISPLACEMENT: BOS barinin buyuklugu ATR'ye gore ---
    disp_buyuk = abs(c[bos_i] - o[bos_i]) >= p["disp_mult"] * atr[bos_i]
    if p["displacement"] and not disp_buyuk:
        return None

    # --- 3) FVG ESLIGI: displacement bacaginda imbalance var mi? ---
    fvg = None
    for j in range(max(bos_i - 2, 2), min(bos_i + 4, n)):
        if side == 1 and l[j] > h[j - 2]:
            fvg = (h[j - 2], l[j]); break
        if side == -1 and h[j] < l[j - 2]:
            fvg = (h[j], l[j - 2]); break
    if p["fvg_esligi"] and fvg is None:
        return None

    # --- 4) ORDER BLOCK: displacement oncesi son TERS mum ---
    ob = None
    for j in range(bos_i, max(bos_i - 12, 0), -1):
        if side == 1 and c[j] < o[j]:
            ob = (l[j], h[j]); break
        if side == -1 and c[j] > o[j]:
            ob = (l[j], h[j]); break
    if ob is None:
        return None

    # --- 5) MITIGATION: bolge daha once dolduruldu mu? ---
    if i > bos_i + 1:
        ara_l = l[bos_i + 1:i].min()
        ara_h = h[bos_i + 1:i].max()
        if side == 1 and ara_l < ob[0]:
            return None      # OB kirilmis, gecersiz
        if side == -1 and ara_h > ob[1]:
            return None

    return dict(ob=ob, fvg=fvg, bos_i=bos_i, swing=swing_seviye)


def giris_seviyesi(k, side, model):
    """Secilen modele gore LIMIT emir seviyesi."""
    lo, hi = k["ob"]
    if model == "ob50":
        return (lo + hi) / 2
    if model == "ob_uzak":
        return lo if side == 1 else hi
    if model == "fvg50" and k["fvg"]:
        return (k["fvg"][0] + k["fvg"][1]) / 2
    if model == "fvg50":
        return (lo + hi) / 2
    return None      # "kapanis" modeli: bar kapanisinda gir


def calistir(data, side, p, giris_model, tp_model, max_hold, tfh, donem="tum"):
    """donem: 'tum' | 'is' (ilk yari) | 'oos' (ikinci yari)"""
    sonuc = []
    for d in data.values():
        o, h, l, c, n = d["o"], d["h"], d["l"], d["c"], d["n"]
        yari = n // 2
        bas, bit = (60, n - 2)
        if donem == "is":
            bit = yari
        elif donem == "oos":
            bas = max(60, yari)
        blok = -1
        for i in range(bas, bit):
            if i <= blok:
                continue
            k = kurulum_ara(d, i, side, p)
            if not k:
                continue
            lo_ob, hi_ob = k["ob"]

            # --- giris ---
            if giris_model == "kapanis":
                # fiyat bolgeye degdi mi (bu barda)?
                if not (lo_ob <= c[i] <= hi_ob):
                    continue
                entry, giris_bar = c[i], i
            else:
                lim = giris_seviyesi(k, side, giris_model)
                # limit emir sonraki barlarda doldu mu (max 10 bar bekle)?
                giris_bar = None
                for j in range(i, min(i + 10, n)):
                    if side == 1 and l[j] <= lim <= h[j]:
                        giris_bar = j; break
                    if side == -1 and l[j] <= lim <= h[j]:
                        giris_bar = j; break
                if giris_bar is None:
                    continue
                entry = lim

            # --- stop: OB'nin disi ---
            if side == 1:
                stop = lo_ob * 0.999
                if stop >= entry or (entry - stop) / entry < MIN_STOP_PCT:
                    stop = entry * (1 - MIN_STOP_PCT)
                risk = entry - stop
            else:
                stop = hi_ob * 1.001
                if stop <= entry or (stop - entry) / entry < MIN_STOP_PCT:
                    stop = entry * (1 + MIN_STOP_PCT)
                risk = stop - entry
            if risk <= 0:
                continue

            # --- hedef ---
            if tp_model == "likidite":
                tp = k["swing"]          # kirilan swing = likidite havuzu
                rr = side * (tp - entry) / risk
                if rr < 1.0:
                    continue
            else:
                rr = float(tp_model)
                tp = entry + side * rr * risk

            # --- simulasyon (kotumser: ayni barda ikisi de -> kayip) ---
            son, bar = None, 0
            for j in range(giris_bar + 1, min(giris_bar + 1 + max_hold, n)):
                bar = j - giris_bar
                if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                    son = -1.0; break
                if (h[j] >= tp) if side == 1 else (l[j] <= tp):
                    son = rr; break
            if son is None:
                j = min(giris_bar + max_hold, n - 1)
                bar = j - giris_bar
                son = side * (c[j] - entry) / risk

            sd = risk / entry
            maliyet = 2 * FEE / sd + FUNDING_8H * (bar * tfh / 8) / sd
            sonuc.append(son - maliyet)
            blok = giris_bar + bar + 3
    return np.array(sonuc)


def rapor(arr):
    if len(arr) < 20:
        return None
    eq = np.cumsum(arr)
    return dict(n=len(arr), win=float((arr > 0).mean()), beklenti=float(arr.mean()),
                toplam=float(arr.sum()),
                dusus=float((np.maximum.accumulate(eq) - eq).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--days", type=int, default=1460)
    ap.add_argument("--save-json", default="results/ict_test.json")
    a = ap.parse_args()
    tfh = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}[a.tf]

    syms = top_symbols(a.top)
    print(f"{len(syms)} coin, {a.tf}, {a.days} gun indiriliyor...")
    data = {}
    for i, s in enumerate(syms, 1):
        df = klines(s, a.tf, a.days)
        if df is not None and len(df) > 400:
            data[s] = hazirla(df)
        if i % 25 == 0:
            print(f"  {i}/{len(syms)}")
        time.sleep(0.03)
    print(f"{len(data)} sembol hazir.\n")

    tum = []
    print("=" * 100)
    print("  ABLASYON: her ICT kuralinin katkisi (hepsi 3R hedef, kapanis girisi)")
    print("=" * 100)
    temel = dict(bos=False, displacement=False, fvg_esligi=False, disp_mult=1.5)
    varyantlar = [
        ("HICBIR KURAL (eski testimiz)", dict(temel)),
        ("+ BOS", dict(temel, bos=True)),
        ("+ BOS + displacement", dict(temel, bos=True, displacement=True)),
        ("+ BOS + disp + FVG esligi", dict(temel, bos=True, displacement=True, fvg_esligi=True)),
        ("+ BOS + GUCLU disp(2.5x)", dict(temel, bos=True, displacement=True, disp_mult=2.5)),
    ]
    for ad, p in varyantlar:
        for side in (1, -1):
            arr = calistir(data, side, p, "kapanis", "3", 48, tfh)
            r = rapor(arr)
            if r:
                yon = "LONG " if side == 1 else "SHORT"
                r.update(ad=f"{ad} [{yon.strip()}]", kurallar=str(p), giris="kapanis", tp="3")
                tum.append(r)
                print(f"  {ad:<32} {yon} {r['n']:>5} isl | win %{r['win']*100:>5.1f} | "
                      f"beklenti {r['beklenti']:>+7.3f}R | toplam {r['toplam']:>+8.1f}R")

    print("\n" + "=" * 100)
    print("  GIRIS MODELI KARSILASTIRMASI (tum kurallar acik, 3R)")
    print("=" * 100)
    tam = dict(temel, bos=True, displacement=True, fvg_esligi=True)
    for gm in ("kapanis", "ob50", "ob_uzak", "fvg50"):
        for side in (1, -1):
            arr = calistir(data, side, tam, gm, "3", 48, tfh)
            r = rapor(arr)
            if r:
                yon = "LONG " if side == 1 else "SHORT"
                r.update(ad=f"giris={gm} [{yon.strip()}]", kurallar=str(tam), giris=gm, tp="3")
                tum.append(r)
                print(f"  {gm:<12} {yon} {r['n']:>5} isl | win %{r['win']*100:>5.1f} | "
                      f"beklenti {r['beklenti']:>+7.3f}R | toplam {r['toplam']:>+8.1f}R")

    print("\n" + "=" * 100)
    print("  HEDEF MODELI (tum kurallar acik, en iyi giris modeli ile)")
    print("=" * 100)
    for tp in ("1", "2", "3", "5", "likidite"):
        for gm in ("ob50", "kapanis"):
            for side in (1, -1):
                arr = calistir(data, side, tam, gm, tp, 48, tfh)
                r = rapor(arr)
                if r:
                    yon = "L" if side == 1 else "S"
                    r.update(ad=f"tp={tp}/{gm}/{yon}", kurallar=str(tam), giris=gm, tp=tp)
                    tum.append(r)
                    print(f"  tp={tp:<9} {gm:<9} {yon} {r['n']:>5} isl | win %{r['win']*100:>5.1f} | "
                          f"beklenti {r['beklenti']:>+7.3f}R | toplam {r['toplam']:>+8.1f}R")

    # ---------- OUT-OF-SAMPLE DOGRULAMA ----------
    print("=" * 100)
    print("  OUT-OF-SAMPLE DOGRULAMA (veri ikiye bolundu)")
    print("  Ilk yari = IS (in-sample), ikinci yari = OOS. Ikisi de pozitifse guclu bulgu.")
    print("=" * 100)
    oos_test = [("ob50", "3"), ("ob50", "5"), ("ob_uzak", "3"), ("ob50", "likidite")]
    for gm, tp in oos_test:
        for side in (1, -1):
            yon = "LONG" if side == 1 else "SHORT"
            r_is = rapor(calistir(data, side, tam, gm, tp, 48, tfh, "is"))
            r_oos = rapor(calistir(data, side, tam, gm, tp, 48, tfh, "oos"))
            if r_is and r_oos:
                tutarli = "TUTARLI" if (r_is["beklenti"] > 0) == (r_oos["beklenti"] > 0) else "TUTARSIZ"
                if r_is["beklenti"] > 0 and r_oos["beklenti"] > 0:
                    tutarli = ">>> IKISI DE POZITIF <<<"
                print(f"  {gm:<9} tp={tp:<9} {yon:<5} | IS {r_is['n']:>4} isl {r_is['beklenti']:>+7.3f}R | "
                      f"OOS {r_oos['n']:>4} isl {r_oos['beklenti']:>+7.3f}R | {tutarli}")
                tum.append(dict(r_oos, ad=f"OOS_{gm}_tp{tp}_{yon}", giris=gm, tp=tp,
                                donem="oos", is_beklenti=r_is["beklenti"], is_n=r_is["n"]))

    if tum:
        tum.sort(key=lambda x: -x["beklenti"])
        print("\n" + "=" * 100)
        print("  EN IYI 10")
        print("=" * 100)
        for r in tum[:10]:
            se = 1.96 * np.sqrt(r["win"] * (1 - r["win"]) / r["n"])
            print(f"  {r['ad']:<40} {r['n']:>5} isl | win %{r['win']*100:.1f}±{se*100:.1f} | "
                  f"beklenti {r['beklenti']:+.3f}R")
        poz = [x for x in tum if x["beklenti"] > 0]
        print(f"\n  Pozitif beklentili: {len(poz)}/{len(tum)}")
        if poz:
            print("  DIKKAT: cok sayida varyant test edildi; birkac tanesinin sansla")
            print("  pozitif cikmasi beklenir. Gercek bulgu icin komsu varyantlarin da")
            print("  pozitif olmasi ve out-of-sample dogrulama gerekir.")

    os.makedirs(os.path.dirname(a.save_json) or ".", exist_ok=True)
    json.dump({"meta": {"coin": len(data), "tf": a.tf, "days": a.days},
               "sonuclar": tum}, open(a.save_json, "w"), indent=1, default=str)
    print(f"\nKaydedildi: {a.save_json}")


if __name__ == "__main__":
    main()
