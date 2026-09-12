"""TREND PORTFOYU - KAGIT TAKIP (lab sonucu, 2026-09-13)
Kural (araclar/lab/tsmom.py 'agirliklar' ile birebir):
  sinyal_i = 1 if close_i > MA50_i else 0        (gunluk kapanis, basit ortalama)
  vol_i    = std(gunluk getiri, 30 gun) * sqrt(365), taban 0.10
  w_i      = (1/K) * sinyal_i * min(HEDEF_VOL / vol_i, CAP)    K = sepetteki coin sayisi
  brut = sum(w) > 1 ise olcekle (kaldirac yok)
Sepetler:
  b3       : BTC, ETH, BNB (sabit; gecmise bakarak secilmis, referans)
  yuruyen3 : her ay basi, onceki gunun 90 gunluk ort USDT hacmine gore ilk 3 (ileriye bakma yok)
Islem: sinyal 00:00 UTC kapanisinda, bot 01:10'da kosar (gecikme.py: gun ici gecikme olculemez).
Maliyet: |dw| * %0.1 (spot taker). Al-tut kiyasi ayni sepetlerde.
Backtest beklentisi (2018-2025, saglamlik.py): Sharpe ~1.0-1.2, maxDD ~%19-28, kotu senaryo 2.3-2.9 yil su alti.
UYARI: yatirim tavsiyesi degildir; kagit portfoydur.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE = "https://data-api.binance.vision"
MA = 50
VOL_WIN = 30
VOL_TABAN = 0.10
HEDEF_VOL = float(os.environ.get("TREND_HEDEF_VOL", "0.20"))
CAP = 1.0
MALIYET = 0.001
B3 = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
HARIC = ("PAXG", "XAUT", "EURI", "AEUR", "EUR", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USD1", "TRY", "BRL")


def klines_1d(sym, limit=200):
    r = requests.get(f"{BASE}/api/v3/klines", params={"symbol": sym, "interval": "1d", "limit": limit}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["ts", "o", "h", "l", "c", "v", "ct", "qav", "n", "tb", "tq", "ig"])
    for x in ("o", "h", "l", "c", "qav"):
        df[x] = df[x].astype(float)
    df["dt"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
    return df[df["dt"] <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)   # sadece kapanmis gunler


def gosterge(df):
    """son kapanmis gun icin: fiyat, ma50, vol30, sinyal"""
    c = df["c"].values
    if len(c) < MA + 2:
        return None
    ma = c[-MA:].mean()
    r = c[-VOL_WIN - 1:] / c[-VOL_WIN - 2:-1] - 1
    vol = max(float(np.std(r, ddof=1) * np.sqrt(365)), VOL_TABAN)
    return dict(fiyat=float(c[-1]), ma=float(ma), vol=vol, sinyal=int(c[-1] > ma), gun=str(df["dt"].iloc[-1].date()))


def yuruyen_sepet(onceki, n=3):
    """Ay basinda 90g ort USDT hacmine gore ilk n vadeli coin; ay icinde onceki sepet korunur."""
    bugun = datetime.now(timezone.utc)
    ay = f"{bugun.year}-{bugun.month:02d}"
    if onceki and onceki.get("ay") == ay and onceki.get("liste"):
        return onceki
    try:
        vadeli = {x["symbol"] for x in requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=25).json()["symbols"]
                  if x.get("quoteAsset") == "USDT" and x.get("contractType") == "PERPETUAL" and x.get("status") == "TRADING"}
    except Exception:
        vadeli = None
    t = requests.get(f"{BASE}/api/v3/ticker/24hr", timeout=30).json()
    aday = sorted([(float(x["quoteVolume"]), x["symbol"]) for x in t
                   if x["symbol"].endswith("USDT") and x["symbol"][:-4] not in HARIC
                   and (vadeli is None or x["symbol"] in vadeli)], reverse=True)[:25]
    skor = []
    for _, s in aday:
        try:
            df = klines_1d(s, 95)
            if len(df) >= 90:
                skor.append((float(df["qav"].values[-90:].mean()), s))
        except Exception:
            pass
    skor.sort(reverse=True)
    return {"ay": ay, "liste": [s for _, s in skor[:n]], "secim_gunu": str(bugun.date())}


def agirliklar(gost, liste, tavan=None):
    """tavan: tek-coin agirlik ust siniri (H5: 0.20; fazlasi nakde, yeniden dagitim yok)"""
    K = len([s for s in liste if s in gost])
    w = {}
    for s in liste:
        g = gost.get(s)
        if not g:
            continue
        w[s] = (1.0 / K) * g["sinyal"] * min(HEDEF_VOL / g["vol"], CAP)
    brut = sum(w.values())
    if brut > 1.0:
        w = {k: v / brut for k, v in w.items()}
    if tavan:
        w = {k: min(v, tavan) for k, v in w.items()}
    return {k: round(v, 4) for k, v in w.items()}


def kagit_ilerlet(p, gost, w_yeni, gun):
    """p: {equity, altut, w, son_fiyat, son_gun}. Bir gun ilerletir: onceki agirliklarla getiri, sonra yeni agirlik + maliyet."""
    if p.get("son_gun") == gun:
        return p
    if p.get("son_fiyat"):
        r = {s: gost[s]["fiyat"] / p["son_fiyat"][s] - 1 for s in p["son_fiyat"] if s in gost}
        g = sum(p["w"].get(s, 0) * r[s] for s in r)
        p["equity"] *= 1 + g
        n = len(r)
        if n:
            p["altut"] *= 1 + sum(r.values()) / n
    dw = sum(abs(w_yeni.get(s, 0) - p.get("w", {}).get(s, 0)) for s in set(w_yeni) | set(p.get("w", {})))
    p["equity"] *= 1 - dw * MALIYET
    p["islem"] = p.get("islem", 0) + (1 if dw > 0.02 else 0)
    p.update(w=w_yeni, son_fiyat={s: gost[s]["fiyat"] for s in w_yeni}, son_gun=gun)
    p.setdefault("baslangic", gun)
    # su alti takibi
    p["zirve"] = max(p.get("zirve", 1.0), p["equity"])
    p["dusus"] = round(1 - p["equity"] / p["zirve"], 4)
    return p


def calistir(state):
    """state: trend_state.json sozlugu. Return (state, mesaj)"""
    pf = state.setdefault("portfoy", {})
    ys = yuruyen_sepet(pf.get("yuruyen_sepet"))
    pf["yuruyen_sepet"] = ys
    # H5 (kesif_tavan, 2026-09-13): yuruyen top-5 + tek-coin tavani 0.20 -> gap riski icin genislik + kuyruk emniyeti
    ys5 = yuruyen_sepet(pf.get("yuruyen_sepet5"), n=5)
    pf["yuruyen_sepet5"] = ys5
    semboller = sorted(set(B3) | set(ys["liste"]) | set(ys5["liste"]))
    gost = {}
    for s in semboller:
        try:
            g = gosterge(klines_1d(s, MA + VOL_WIN + 10))
            if g:
                gost[s] = g
        except Exception as e:
            print(f"  [portfoy] {s} veri hatasi: {e}")
    gun = max(g["gun"] for g in gost.values())
    satirlar = [f"\U0001F4BC <b>TREND PORTFOYU</b> (MA{MA}, hedef vol %{HEDEF_VOL*100:.0f}, kaldirac yok, kagit)",
                f"Gun: {gun} | islem saati 01:10 UTC"]
    for ad, liste, key, tavan in (("b3 BTC/ETH/BNB", B3, "b3", None),
                                  (f"yuruyen top-3 ({ys['ay']})", ys["liste"], "yuruyen3", None),
                                  (f"H5 yuruyen top-5 + tavan %20 ({ys5['ay']})", ys5["liste"], "yuruyen5_tavan", 0.20)):
        w = agirliklar(gost, liste, tavan)
        p = kagit_ilerlet(pf.setdefault(key, {"equity": 1.0, "altut": 1.0, "w": {}, "son_fiyat": None}), gost, w, gun)
        satirlar.append(f"\n<b>{ad}</b>  brut %{sum(w.values())*100:.0f}")
        for s in liste:
            g = gost.get(s)
            if not g:
                satirlar.append(f"  {s[:-4]:<6} veri yok"); continue
            isaret = "\U0001F7E2" if g["sinyal"] else "\U0001F534"
            satirlar.append(f"  {isaret} {s[:-4]:<6} w=%{w.get(s,0)*100:4.0f} | MA{MA} %{(g['fiyat']/g['ma']-1)*100:+5.1f} | vol %{g['vol']*100:.0f}")
        satirlar.append(f"  kagit {(p['equity']-1)*100:+.1f}% | al-tut {(p['altut']-1)*100:+.1f}% | dusus %{p['dusus']*100:.1f} | {p['baslangic']}'den beri")
    satirlar.append("\n<i>Beklenti (2018-25 backtest): Sharpe ~1.0-1.2, maxDD ~%19-28; kotu senaryo 2-3 yil su alti. "
                    "Kanit: RAPOR_2026-09-12.md, araclar/lab. Yatirim tavsiyesi degildir.</i>")
    return state, "\n".join(satirlar)


if __name__ == "__main__":
    st = json.load(open("trend_state.json")) if os.path.exists("trend_state.json") else {}
    st, msg = calistir(st)
    print(msg)
    json.dump(st, open("trend_state.json", "w"), indent=1)
