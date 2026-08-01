#!/usr/bin/env python3
"""
MOMENTUM BACKTEST  --  Trend Takibi + Kesitsel Momentum + Kiyas Noktalari
=========================================================================
Test edilen 4 yaklasim:

  1) TREND TAKIBI (time-series momentum)
     Kural: fiyat N gunluk ortalamasinin USTUNDEyse tut, ALTINDAysa nakitte kal.
     Literatur: Moskowitz-Ooi-Pedersen (2012), 58 varlik / 25 yil.

  2) KESITSEL MOMENTUM (cross-sectional)
     Kural: her yenileme gununde son K gunde en cok yukselen ilk N coini tut.
     Literatur: Liu & Tsyvinski (2021), kripto momentum.

  3) BTC AL-TUT  (kiyas)
  4) BTC DCA / duzenli alim  (kiyas)

Aktif bir strateji, 3 ve 4'u GECEMIYORSA anlamsizdir. Rapor bunu acikca gosterir.

Calistirma:
    python momentum_backtest.py --top 100 --days 1460 --capital 4000000

UYARI: Bu bir yatirim tavsiyesi degildir. Gecmis performans gelecegi
garanti etmez. Sonuclar kayma (slippage) ve vergi icermez.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://data-api.binance.vision"
FEE = 0.001          # gidis-donus komisyon (%0.1)


# ---------------------------------------------------------------- veri
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
    tf_ms = {"1d": 86400, "4h": 14400}[interval] * 1000
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
    df["c"] = df["c"].astype(float)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").sort_values("dt").reset_index(drop=True)


# ------------------------------------------------------- 1) trend takibi
def trend_following(px, ma_days=100):
    """Fiyat MA'nin ustundeyse yatirimda, altindaysa nakitte.
    Sinyal BIR GUN GECIKMELI uygulanir (ileriye bakma yok)."""
    ma = px.rolling(ma_days).mean()
    pozisyon = (px > ma).shift(1).fillna(False)          # dun karar -> bugun uygula
    getiri = px.pct_change().fillna(0)
    islem = pozisyon.astype(int).diff().abs().fillna(0)  # giris/cikis sayisi
    net = getiri * pozisyon - islem * FEE
    return (1 + net).cumprod(), int(islem.sum())


# --------------------------------------------- 2) kesitsel momentum
def cross_sectional(fiyatlar, geri=30, tut=10, yenile=7):
    """Her `yenile` gunde bir, son `geri` gunun en iyi `tut` coinini esit agirlikla tut."""
    getiri = fiyatlar.pct_change().fillna(0)
    mom = fiyatlar.pct_change(geri)
    equity = [1.0]
    tarihler = fiyatlar.index
    secili = []
    islem = 0
    for i in range(geri + 1, len(tarihler)):
        if (i - geri - 1) % yenile == 0:
            sira = mom.iloc[i - 1].dropna().sort_values(ascending=False)
            yeni = list(sira.head(tut).index)
            degisen = len(set(yeni) ^ set(secili))
            islem += degisen
            maliyet = (degisen / max(len(yeni), 1)) * FEE if yeni else 0
            secili = yeni
        else:
            maliyet = 0
        gun = getiri.iloc[i][secili].mean() if secili else 0.0
        equity.append(equity[-1] * (1 + gun - maliyet))
    return pd.Series(equity, index=tarihler[geri:]), islem


# ------------------------------------------------------ 3-4) kiyaslar
def al_tut(px):
    return px / px.iloc[0]


def dca(px, periyot=7):
    """Her `periyot` gunde sabit tutar alim. Return: equity (yatirilan sermayeye gore)."""
    alim_gunleri = range(0, len(px), periyot)
    adet, yatirilan = 0.0, 0.0
    seri = []
    for i in range(len(px)):
        if i in alim_gunleri:
            adet += 1.0 / px.iloc[i]
            yatirilan += 1.0
        seri.append(adet * px.iloc[i] / max(yatirilan, 1e-9))
    return pd.Series(seri, index=px.index)


# ------------------------------------------------------------- rapor
def olcut(eq):
    if len(eq) < 2:
        return None
    toplam = float(eq.iloc[-1] / eq.iloc[0] - 1)
    yil = len(eq) / 365
    yillik = (1 + toplam) ** (1 / yil) - 1 if yil > 0 else 0
    dd = float((1 - eq / eq.cummax()).max())
    g = eq.pct_change().dropna()
    sharpe = float(g.mean() / g.std() * np.sqrt(365)) if g.std() > 0 else 0
    return dict(toplam=toplam, yillik=yillik, max_dusus=dd, sharpe=sharpe)


def yaz(ad, eq, sermaye, islem=None):
    m = olcut(eq)
    if not m:
        print(f"  {ad}: veri yok")
        return None
    son = sermaye * (1 + m["toplam"])
    ek = f" | {islem} islem" if islem is not None else ""
    print(f"  {ad:<28} {sermaye:>12,.0f} -> {son:>12,.0f} TL  "
          f"({m['toplam']*100:+7.1f}% | yillik {m['yillik']*100:+6.1f}% | "
          f"max dusus -{m['max_dusus']*100:.0f}% | sharpe {m['sharpe']:.2f}){ek}")
    return {**m, "son_para": son, "ad": ad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--days", type=int, default=1460)
    ap.add_argument("--capital", type=float, default=4_000_000)
    ap.add_argument("--save-json", default="results/momentum.json")
    a = ap.parse_args()

    print(f"Veri indiriliyor: ilk {a.top} coin, {a.days} gun (gunluk mum)")
    syms = top_symbols(a.top)
    seri = {}
    for i, s in enumerate(syms, 1):
        df = klines(s, "1d", a.days)
        if df is not None and len(df) > 150:
            seri[s] = df.set_index("dt")["c"]
        if i % 25 == 0:
            print(f"  {i}/{len(syms)}")
        time.sleep(0.03)
    if "BTCUSDT" not in seri:
        sys.exit("BTC verisi alinamadi.")

    fiyat = pd.DataFrame(seri).sort_index()
    fiyat = fiyat[fiyat.index >= fiyat["BTCUSDT"].first_valid_index()]
    btc = fiyat["BTCUSDT"].dropna()
    print(f"{len(fiyat.columns)} coin | {fiyat.index[0].date()} -> {fiyat.index[-1].date()}")

    print(f"\n{'=' * 100}\n  SONUCLAR  ({a.capital:,.0f} TL ile)\n{'=' * 100}")
    sonuc = []

    print("\n  --- KIYAS NOKTALARI (hicbir sey yapmamak) ---")
    sonuc.append(yaz("BTC al-tut", al_tut(btc), a.capital))
    sonuc.append(yaz("BTC DCA (haftalik alim)", dca(btc), a.capital))

    print("\n  --- 1) TREND TAKIBI (BTC, fiyat > MA ise tut) ---")
    for ma in (50, 100, 200):
        eq, n = trend_following(btc, ma)
        sonuc.append(yaz(f"BTC trend MA{ma}", eq.dropna(), a.capital, n))

    print("\n  --- 1b) TREND TAKIBI HER COINDE (MA50) ---")
    print("      Not: MA50 daha once SADECE BTCde test edilmisti. Bu bolum")
    print("      altcoinlerde de calisip calismadigini olcer.")
    tf_sonuc, kotu, iyi = [], 0, 0
    for sym in fiyat.columns:
        px = fiyat[sym].dropna()
        if len(px) < 250:
            continue
        eq, _ = trend_following(px, 50)
        eq = eq.dropna()
        if len(eq) < 100:
            continue
        m_tf, m_bh = olcut(eq), olcut(al_tut(px))
        if not m_tf or not m_bh:
            continue
        fark = m_tf["toplam"] - m_bh["toplam"]
        tf_sonuc.append((sym, m_tf, m_bh, fark))
        iyi += fark > 0
        kotu += fark <= 0
    if tf_sonuc:
        o_tf = np.mean([x[1]["toplam"] for x in tf_sonuc])
        o_bh = np.mean([x[2]["toplam"] for x in tf_sonuc])
        d_tf = np.mean([x[1]["max_dusus"] for x in tf_sonuc])
        d_bh = np.mean([x[2]["max_dusus"] for x in tf_sonuc])
        print(f"      {len(tf_sonuc)} coin | MA50 al-tutu {iyi} coinde GECTI, "
              f"{kotu} coinde GECEMEDI (%{iyi/len(tf_sonuc)*100:.0f})")
        print(f"      ort getiri : MA50 {o_tf*100:+.0f}%  vs  al-tut {o_bh*100:+.0f}%")
        print(f"      ort dusus  : MA50 -{d_tf*100:.0f}%  vs  al-tut -{d_bh*100:.0f}%")
        print("      -> MA50nin asil faydasi DUSUSU AZALTMAK; getiri farki ikincildir.")
        tf_sonuc.sort(key=lambda x: -x[3])
        print("      en iyi 5:", ", ".join(f"{x[0][:-4]}({x[3]*100:+.0f}%)" for x in tf_sonuc[:5]))
        print("      en kotu 5:", ", ".join(f"{x[0][:-4]}({x[3]*100:+.0f}%)" for x in tf_sonuc[-5:]))
    print("\n  --- 2) KESITSEL MOMENTUM (en cok yukselen N coin) ---")
    for geri, tut, yenile in ((30, 10, 7), (30, 5, 7), (90, 10, 30), (14, 10, 7)):
        eq, n = cross_sectional(fiyat, geri, tut, yenile)
        sonuc.append(yaz(f"Momentum {geri}g/ilk{tut}/{yenile}g", eq, a.capital, n))

    sonuc = [s for s in sonuc if s]
    sonuc.sort(key=lambda x: -x["son_para"])
    print(f"\n{'=' * 100}\n  SIRALAMA (en cok buyuten)\n{'=' * 100}")
    for i, s in enumerate(sonuc, 1):
        print(f"  {i:2d}. {s['ad']:<28} {s['son_para']:>12,.0f} TL   "
              f"max dusus -{s['max_dusus']*100:.0f}%")

    en_iyi = sonuc[0]
    kiyas = [s for s in sonuc if "al-tut" in s["ad"] or "DCA" in s["ad"]]
    print(f"\n  EN IYI: {en_iyi['ad']}")
    if kiyas and en_iyi["ad"] not in [k["ad"] for k in kiyas]:
        en_iyi_kiyas = max(kiyas, key=lambda x: x["son_para"])
        fark = en_iyi["son_para"] - en_iyi_kiyas["son_para"]
        print(f"  Kiyasa gore fark: {fark:+,.0f} TL "
              f"({en_iyi_kiyas['ad']} ile karsilastirildiginda)")
        if fark <= 0:
            print("  -> Aktif strateji kiyasi GECEMEDI. Ugrasmaya degmez.")
    else:
        print("  -> Hicbir aktif strateji basit kiyasi gecemedi.")

    os.makedirs(os.path.dirname(a.save_json) or ".", exist_ok=True)
    with open(a.save_json, "w") as f:
        json.dump({"meta": {"coin": len(fiyat.columns), "gun": a.days,
                            "sermaye": a.capital,
                            "baslangic": str(fiyat.index[0].date()),
                            "bitis": str(fiyat.index[-1].date())},
                   "sonuclar": sonuc}, f, indent=1, default=str)
    print(f"\nKaydedildi: {a.save_json}")
    print("\nUYARI: Yatirim tavsiyesi degildir. Kayma/vergi dahil degildir.\n"
          "Max dusus sutununa dikkat: -%50 demek, paranin yarisini gecici olarak\n"
          "kaybetmeye dayanmak demektir. Buna dayanamayacaksan o strateji sana uygun degil.")


if __name__ == "__main__":
    main()
