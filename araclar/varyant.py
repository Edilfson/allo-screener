"""Parametre varyantlarini kosar, IS(ilk yil)/OOS(ikinci yil) net R ile karsilastirir.
ON KAYIT: her varyant icin 'sağlam' = IS ve OOS'ta ayni isaret + tum-veri GA sifiri disliyor.
Coklu test: ~20 varyant -> Bonferroni benzeri esik: GA %99 kullan."""
import json, os, sys, subprocess
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
VARYANT = [
    ("temel", ""),
    ("tp=2", "TP_R=2"), ("tp=3", "TP_R=3"), ("tp=8", "TP_R=8"),
    ("disp=1.0", "DISP_MULT=1.0"), ("disp=2.0", "DISP_MULT=2.0"), ("disp=2.5", "DISP_MULT=2.5"),
    ("mesafe=0.5", "MAX_ENTRY_DIST_R=0.5"), ("mesafe=1.0", "MAX_ENTRY_DIST_R=1.0"), ("mesafe=3", "MAX_ENTRY_DIST_R=3"),
    ("bos_yas=8", "MAX_BOS_YAS=8"), ("bos_yas=40", "MAX_BOS_YAS=40"),
    ("minstop=1%", "MIN_STOP_PCT=0.01"), ("minstop=1.5%", "MIN_STOP_PCT=0.015"),
    ("maxstop=2%", "MAX_STOP_PCT=0.02"),
    ("bekle=48s", "PENDING_H=48"), ("bekle=8s", "PENDING_H=8"),
    ("tut=3g", "MAX_DAYS=3"), ("tut=14g", "MAX_DAYS=14"),
    ("swing_k=5", "SWING_K=5"),
]
tfs = sys.argv[1] if len(sys.argv) > 1 else "1h,4h"

def ga(x, n=3000, q=.005):
    x = np.asarray(x); m = np.sort([x[np.random.randint(0, len(x), len(x))].mean() for _ in range(n)])
    return m[int(n * q)], m[int(n * (1 - q))]

print(f"{'varyant':<14}{'sinyal':>7}{'dolan':>7}{'net/isl':>9}{'GA99':>18}{'IS':>8}{'OOS':>8}  saglam?")
for ad, s in VARYANT:
    out = f"v_{ad.replace('=','_').replace('%','p').replace('.','_')}.json"
    if not os.path.exists(os.path.join(D, out)):
        subprocess.run([PY, os.path.join(D, "bt.py"), "--tfs", tfs, "--out", out, "--set", s], check=True,
                       capture_output=True)
    J = json.load(open(os.path.join(D, out))); T = J["trades"]; F = [t for t in T if t["res"] != "cancelled"]
    if len(F) < 30:
        print(f"{ad:<14}{len(T):>7}{len(F):>7}   (az)"); continue
    ts = sorted(t["t"] for t in F); mid = ts[len(ts) // 2]
    x = np.array([t["net"] for t in F]); a = np.array([t["net"] for t in F if t["t"] < mid]); b = np.array([t["net"] for t in F if t["t"] >= mid])
    lo, hi = ga(x)
    sag = "EVET" if (lo > 0 and a.mean() > 0 and b.mean() > 0) else ("kotu" if hi < 0 else "-")
    print(f"{ad:<14}{len(T):>7}{len(F):>7}{x.mean():>+9.3f}   [{lo:+.3f},{hi:+.3f}]{a.mean():>+8.3f}{b.mean():>+8.3f}  {sag}")
