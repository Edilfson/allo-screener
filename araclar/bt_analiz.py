"""bt.py ciktisini istatistiksel disiplinle raporlar: bootstrap GA, IS/OOS, ceyrek tutarliligi."""
import json, sys, os, random
from datetime import datetime, timezone
import numpy as np
random.seed(11)
D = os.path.dirname(os.path.abspath(__file__))
f = sys.argv[1] if len(sys.argv) > 1 else "trades_base.json"
J = json.load(open(os.path.join(D, f))); T = J["trades"]
F = [t for t in T if t["res"] != "cancelled"]
print(f"== {f} | P farklari: { {k:v for k,v in J['P'].items()} }")
print(f"sinyal {len(T)} | dolan {len(F)} (%{len(F)/max(1,len(T))*100:.0f}) | "
      f"tarih {datetime.fromtimestamp(min(t['t'] for t in T)/1e3, timezone.utc):%Y-%m-%d} -> "
      f"{datetime.fromtimestamp(max(t['t'] for t in T)/1e3, timezone.utc):%Y-%m-%d}")

def ga(x, n=4000):
    if len(x) < 5: return float("nan"), float("nan")
    x = np.asarray(x); m = np.sort([x[np.random.randint(0, len(x), len(x))].mean() for _ in range(n)])
    return m[int(n * .025)], m[int(n * .975)]

def satir(ad, g, key="net"):
    x = [t[key] for t in g]
    if not x: return
    lo, hi = ga(x); w = np.mean([v > 0 for v in x]); m = np.mean(x)
    isaret = "**" if lo > 0 else ("--" if hi < 0 else "  ")
    print(f"  {isaret}{ad:<28} n={len(x):>5} ort={m:+.3f} GA[{lo:+.3f},{hi:+.3f}] win=%{w*100:4.1f} top={sum(x):+8.1f}")

def yari(g):
    ts = sorted(t["t"] for t in g); mid = ts[len(ts) // 2] if ts else 0
    return [t for t in g if t["t"] < mid], [t for t in g if t["t"] >= mid]

def ceyrek(t):
    d = datetime.fromtimestamp(t["t"] / 1e3, timezone.utc); return f"{d.year}Q{(d.month-1)//3+1}"

def blok(ad, fn, grp=F, key="net", ceyrekler=False):
    print(f"\n== {ad} [{key}] ==")
    for k in sorted({fn(t) for t in grp}, key=str):
        g = [t for t in grp if fn(t) == k]
        satir(str(k), g, key)
        a, b = yari(g)
        satir(f"   1.yari", a, key); satir(f"   2.yari", b, key)
        if ceyrekler:
            qs = sorted({ceyrek(t) for t in g})
            print("     ceyrekler: " + " | ".join(f"{q} {np.mean([t[key] for t in g if ceyrek(t)==q]):+.2f}(n{sum(ceyrek(t)==q for t in g)})" for q in qs))

satir("TUMU brut", F, "r"); satir("TUMU net", F, "net")
a, b = yari(F); satir("  1.yari net", a); satir("  2.yari net", b)
blok("DILIM", lambda t: t["tf"], ceyrekler=True)
blok("DILIM x YON", lambda t: (t["tf"], "L" if t["side"] == 1 else "S"))
uy = lambda t: ("trend_yonunde" if t["ct"] * t["side"] > 0 else "trende_karsi" if t["ct"] != 0 else "kararsiz")
blok("H1: YON x 1D TREND", uy, ceyrekler=True)
blok("H1 dilim bazinda", lambda t: (t["tf"], uy(t)), [t for t in F if t["tf"] in ("1h", "4h")])
blok("BTC rejimi x yon", lambda t: ("btc_boga" if t["bt"] >= 0 else "btc_ayi", "L" if t["side"] == 1 else "S"))
band = lambda t: "a<0.8" if t["sp"] < .008 else "b0.8-1.2" if t["sp"] < .012 else "c1.2-2" if t["sp"] < .02 else "d2-5"
blok("STOP BANDI", band)
blok("GIRIS MESAFESI (R)", lambda t: "a<0.5" if t["mes"] < .5 else "b0.5-1" if t["mes"] < 1 else "c1-1.5")
blok("BOS YASI", lambda t: "a<=5" if t["yas"] <= 5 else "b6-10" if t["yas"] <= 10 else "c11-20")
