"""Olgun evren (>=3 yil gecmisi olan coinler): trend takibi vs al-tut, coin bazinda."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trend_bt as T
from datetime import datetime, timezone
syms, gun, O, C = T.yukle()
gecmis = (~np.isnan(C)).sum(0)
sec = gecmis >= 3 * 365
print(f"olgun coin: {sec.sum()}/{len(syms)}")
O2, C2 = O[:, sec], C[:, sec]; s2 = [s for s, k in zip(syms, sec) if k]
tarih = np.array([datetime.fromtimestamp(x / 1e3, timezone.utc) for x in gun]); yil = np.array([d.year for d in tarih])
def rapor(ad, sig, funding=0.0):
    eq, g = T.kos(O2, C2, sig, funding=funding); mt = T.metrik(eq, g, gun)
    yl = " ".join(f"{y}:{(np.prod(1+g[yil[1:-1]==y])-1)*100:+.0f}%" for y in sorted(set(yil)) if (yil[1:-1]==y).sum() > 30)
    print(f"  {ad:<26} toplam {mt['toplam']*100:+7.0f}%  CAGR {mt['cagr']*100:+6.1f}%  maxDD {mt['maxdd']*100:5.1f}%  Sharpe {mt['sharpe']:5.2f} | {yl}")
print("== OLGUN EVREN ==")
rapor("esit-agirlik AL-TUT", np.where(np.isnan(C2), 0, 1).astype(float))
for N in (20, 50, 100, 200):
    M = T.ma(C2, N); s = np.where(C2 > M, 1.0, 0.0); s[np.isnan(M)] = 0
    rapor(f"MA{N} long-only", s)
j = syms.index("BTCUSDT"); Mb = T.ma(C[:, j:j+1], 50)[:, 0]; rej = (C[:, j] > Mb).astype(float)
for N in (50, 100):
    M = T.ma(C2, N); s = np.where(C2 > M, 1.0, 0.0); s[np.isnan(M)] = 0
    rapor(f"MA{N} x BTC>MA50 rejimi", s * rej[:, None])
rapor("AL-TUT x BTC>MA50 rejimi", np.where(np.isnan(C2), 0, 1).astype(float) * rej[:, None])

print("\n== COIN BAZINDA: MA100 long-only vs al-tut (olgun evren) ==")
kaz = 0; ddk = 0; rows = []
for k, s in enumerate(s2):
    o, c = O2[:, k], C2[:, k]; ok = ~np.isnan(c); o, c = o[ok], c[ok]
    m = np.convolve(c, np.ones(100) / 100, "full")[:len(c)]; m[:99] = np.nan
    sig = np.where(c > m, 1.0, 0.0); sig[np.isnan(m)] = 0
    e1, g1 = T.kos(o[:, None], c[:, None], sig[:, None]); e0, g0 = T.kos(o[:, None], c[:, None], np.ones((len(c), 1)))
    m1, m0 = T.metrik(e1, g1, None), T.metrik(e0, g0, None)
    kaz += m1["cagr"] > m0["cagr"]; ddk += m1["maxdd"] < m0["maxdd"]
    rows.append((s, m0["cagr"], m1["cagr"], m0["maxdd"], m1["maxdd"]))
print(f"  CAGR'da al-tutu gecen: {kaz}/{len(s2)} | maxDD'yi dusuren: {ddk}/{len(s2)}")
print(f"  medyan CAGR al-tut {np.median([r[1] for r in rows])*100:+.1f}% vs MA100 {np.median([r[2] for r in rows])*100:+.1f}% | "
      f"medyan maxDD {np.median([r[3] for r in rows])*100:.0f}% vs {np.median([r[4] for r in rows])*100:.0f}%")
for r in sorted(rows, key=lambda x: -x[2])[:8]: print(f"    {r[0]:<12} altut {r[1]*100:+6.1f}%/{r[3]*100:3.0f}%  MA100 {r[2]*100:+6.1f}%/{r[4]*100:3.0f}%")
