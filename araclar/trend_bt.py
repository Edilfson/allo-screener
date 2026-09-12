"""TREND TAKIBI (time-series momentum) backtesti - gunluk, 100 coin, 4 yil.
Kural: t gununun kapanisinda sinyal, t+1 gununun ACILISINDA islem (ileriye bakma yok).
Sinyal: close > MA(N)  (long) | close < MA(N) (short, sadece L/S modunda)
Portfoy: sinyalde olan coinlere esit agirlik, kalan nakit (%0 faiz). Gunluk yeniden dengeleme.
Maliyet: agirlik degisiminin |delta| * MALIYET (taker+kayma, tek yon).
Kiyas: ayni evrenin esit-agirlik AL-TUT'u ve BTC al-tut (hayatta-kalma yanliligi her ikisinde de var).
"""
import os, json, sys
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data1d")
MALIYET = 0.0007      # tek yon: taker %0.05 + kayma %0.02 (vadeli); spot icin %0.1 kullan
FUNDING_G = 0.0001 * 3  # vadeli funding gun basi (%0.01 x 3), long ve short icin simetrik varsayim

def yukle():
    syms = [f[:-4] for f in os.listdir(D) if f.endswith(".npy")]
    seri = {s: np.load(os.path.join(D, s + ".npy")) for s in syms}
    gunler = sorted({int(x) for a in seri.values() for x in a[:, 0]})
    gi = {g: i for i, g in enumerate(gunler)}
    n, m = len(gunler), len(syms)
    O = np.full((n, m), np.nan); C = np.full((n, m), np.nan)
    for j, s in enumerate(syms):
        a = seri[s]
        idx = [gi[int(x)] for x in a[:, 0]]
        O[idx, j] = a[:, 1]; C[idx, j] = a[:, 4]
    return syms, np.array(gunler), O, C

def ma(C, N):
    out = np.full_like(C, np.nan)
    cs = np.nancumsum(np.nan_to_num(C), axis=0)
    cnt = np.cumsum(~np.isnan(C), axis=0)
    for i in range(N - 1, len(C)):
        s = cs[i] - (cs[i - N] if i >= N else 0); k = cnt[i] - (cnt[i - N] if i >= N else 0)
        out[i] = np.where(k == N, s / N, np.nan)
    return out

def kos(O, C, sinyal, maliyet=MALIYET, funding=0.0):
    """sinyal[t,j] in {-1,0,+1}, t kapanisinda bilinir; t+1 acilista uygulanir, t+2 acilisa kadar tutulur.
    Gunluk getiri: open[t+1]->open[t+2]. Esit agirlik (aktif sinyal sayisina bolunur)."""
    n, m = C.shape
    eq = [1.0]; w_prev = np.zeros(m); getiriler = []
    for t in range(n - 2):
        s = np.nan_to_num(sinyal[t]); s[np.isnan(O[t + 1]) | np.isnan(O[t + 2])] = 0
        k = (s != 0).sum()
        w = s / k if k else np.zeros(m)
        r = np.nan_to_num((O[t + 2] - O[t + 1]) / O[t + 1])
        r = np.clip(r, -0.95, 5.0)
        g = (w * r).sum() - np.abs(w - w_prev).sum() * maliyet - np.abs(w).sum() * funding
        getiriler.append(g); eq.append(eq[-1] * (1 + g)); w_prev = w * (1 + r) / (1 + g) if k else w
    return np.array(eq), np.array(getiriler)

def metrik(eq, g, gun):
    yil = len(g) / 365
    cagr = eq[-1] ** (1 / yil) - 1
    dd = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)
    sh = g.mean() / g.std() * np.sqrt(365) if g.std() > 0 else 0
    return dict(toplam=eq[-1] - 1, cagr=cagr, maxdd=dd.max(), sharpe=sh)

def yillik(g, gun):
    out = {}
    for y in sorted({int(str(x)[:4]) for x in gun}):
        pass
    return out

if __name__ == "__main__":
    syms, gun, O, C = yukle()
    from datetime import datetime, timezone
    tarih = np.array([datetime.fromtimestamp(x / 1e3, timezone.utc) for x in gun])
    yil = np.array([d.year for d in tarih])
    print(f"{len(syms)} coin | {tarih[0]:%Y-%m-%d} -> {tarih[-1]:%Y-%m-%d} | {len(gun)} gun")
    print(f"maliyet tek yon %{MALIYET*100:.2f}\n")

    def rapor(ad, sinyal, funding=0.0):
        eq, g = kos(O, C, sinyal, funding=funding)
        mt = metrik(eq, g, gun)
        yl = " ".join(f"{y}:{(np.prod(1+g[yil[1:-1]==y])-1)*100:+.0f}%" for y in sorted(set(yil)) if (yil[1:-1]==y).sum() > 30)
        print(f"  {ad:<26} toplam {mt['toplam']*100:+7.0f}%  CAGR {mt['cagr']*100:+6.1f}%  maxDD {mt['maxdd']*100:5.1f}%  Sharpe {mt['sharpe']:5.2f} | {yl}")
        return mt, g

    print("== KIYAS ==")
    bh = np.where(np.isnan(C), 0, 1).astype(float)
    rapor("Evren esit-agirlik AL-TUT", bh)
    j = syms.index("BTCUSDT"); btc = np.zeros_like(C); btc[:, j] = 1
    rapor("BTC AL-TUT", btc)

    print("\n== TREND TAKIBI, LONG-ONLY (fiyat > MA -> tut, degilse nakit) ==")
    for N in (20, 50, 100, 200):
        M = ma(C, N); s = np.where(C > M, 1.0, 0.0); s[np.isnan(M)] = 0
        rapor(f"MA{N} long-only", s)
    print("\n== TREND TAKIBI, LONG/SHORT (vadeli, funding dahil) ==")
    for N in (20, 50, 100, 200):
        M = ma(C, N); s = np.where(C > M, 1.0, -1.0); s[np.isnan(M)] = 0
        rapor(f"MA{N} long/short", s, funding=FUNDING_G)
    print("\n== SADECE BTC, MA50 long-only (trend_bot.py'nin iddiasi) ==")
    M = ma(C, 50); s = np.zeros_like(C); s[:, j] = np.where(C[:, j] > M[:, j], 1.0, 0.0); s[np.isnan(M[:, j]), j] = 0
    rapor("BTC MA50 long-only", s)
    print("\n== BTC REJIM FILTRESI: BTC > MA50 iken evren long-only MA50, degilse nakit ==")
    M = ma(C, 50); s = np.where(C > M, 1.0, 0.0); s[np.isnan(M)] = 0
    rej = (C[:, j] > M[:, j]).astype(float); s2 = s * rej[:, None]
    rapor("evren MA50 x BTC rejimi", s2)
    print("\n== K-GUNLUK GETIRI ISARETI (MA'ya alternatif tanim - komsu varyant) ==")
    for K in (60, 120, 180):
        s = np.zeros_like(C); s[K:] = np.where(C[K:] > C[:-K], 1.0, 0.0); s[np.isnan(C)] = 0
        s[K:][np.isnan(C[:-K])] = 0
        rapor(f"getiri{K}>0 long-only", s)
