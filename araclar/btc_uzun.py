"""BTC/ETH/BNB 2017'den beri gunluk: MA trend takibi komsu varyantlar + yil bazinda OOS."""
import os, time, json
import numpy as np, requests
from datetime import datetime, timezone
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_uzun"); os.makedirs(D, exist_ok=True)
MALIYET = 0.001   # spot tek yon %0.1 (kotumser)

def indir(sym):
    out = os.path.join(D, sym + ".npy")
    if os.path.exists(out): return np.load(out)
    since, rows = 1500000000000, []
    while True:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": sym, "interval": "1d", "startTime": since, "limit": 1000}, timeout=30)
        b = r.json()
        if not b: break
        rows += [[x[0], float(x[1]), float(x[4])] for x in b]
        since = b[-1][0] + 86_400_000
        if len(b) < 1000: break
        time.sleep(0.2)
    a = np.array(rows[:-1]); np.save(out, a); return a

def kos(o, c, sig):
    """sig[t] kapanista bilinir; t+1 acilista uygula; getiri open[t+1]->open[t+2]"""
    eq = [1.0]; g = []; prev = 0.0
    for t in range(len(c) - 2):
        w = sig[t]; r = (o[t + 2] - o[t + 1]) / o[t + 1]
        x = w * r - abs(w - prev) * MALIYET
        g.append(x); eq.append(eq[-1] * (1 + x)); prev = w
    return np.array(eq), np.array(g)

def met(eq, g):
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    return eq[-1] - 1, eq[-1] ** (365 / len(g)) - 1, dd, g.mean() / g.std() * np.sqrt(365)

for sym in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
    a = indir(sym); ot, o, c = a[:, 0], a[:, 1], a[:, 2]
    yil = np.array([datetime.fromtimestamp(x / 1e3, timezone.utc).year for x in ot])[1:-1]
    yillar = sorted(set(yil))
    print(f"\n==== {sym}  {datetime.fromtimestamp(ot[0]/1e3, timezone.utc):%Y-%m-%d} -> {datetime.fromtimestamp(ot[-1]/1e3, timezone.utc):%Y-%m-%d}  ({len(c)} gun, maliyet %{MALIYET*100:.1f}) ====")
    print(f"  {'strateji':<14}{'toplam':>10}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8} | " + " ".join(f"{y:>6}" for y in yillar))
    def rapor(ad, sig):
        eq, g = kos(o, c, sig); t, cg, dd, sh = met(eq, g)
        yl = " ".join(f"{(np.prod(1+g[yil==y])-1)*100:+5.0f}%" for y in yillar)
        print(f"  {ad:<14}{t*100:>+9.0f}%{cg*100:>+7.1f}%{dd*100:>7.1f}%{sh:>8.2f} | {yl}")
    rapor("AL-TUT", np.ones(len(c)))
    for N in (10, 20, 30, 50, 75, 100, 150, 200):
        m = np.convolve(c, np.ones(N) / N, "full")[:len(c)]; m[:N - 1] = np.nan
        s = np.where(c > m, 1.0, 0.0); s[np.isnan(m)] = 0
        rapor(f"MA{N}", s)
    for K in (30, 60, 90, 180):
        s = np.zeros(len(c)); s[K:] = (c[K:] > c[:-K]).astype(float)
        rapor(f"getiri{K}>0", s)
    # ILK YARI / IKINCI YARI (2017-2021 vs 2022-2026): MA50 ve komsulari
    h = len(c) // 2
    print("  -- yarilar (CAGR / maxDD): IS=ilk yari, OOS=ikinci yari --")
    for N in (20, 50, 100, 200):
        m = np.convolve(c, np.ones(N) / N, "full")[:len(c)]; m[:N - 1] = np.nan
        s = np.where(c > m, 1.0, 0.0); s[np.isnan(m)] = 0
        r1 = met(*kos(o[:h], c[:h], s[:h])); r2 = met(*kos(o[h:], c[h:], s[h:]))
        b1 = met(*kos(o[:h], c[:h], np.ones(h))); b2 = met(*kos(o[h:], c[h:], np.ones(len(c) - h)))
        print(f"    MA{N:<4} IS {r1[1]*100:+6.1f}%/{r1[2]*100:4.0f}%  (altut {b1[1]*100:+6.1f}%/{b1[2]*100:4.0f}%) | "
              f"OOS {r2[1]*100:+6.1f}%/{r2[2]*100:4.0f}%  (altut {b2[1]*100:+6.1f}%/{b2[2]*100:4.0f}%)")
