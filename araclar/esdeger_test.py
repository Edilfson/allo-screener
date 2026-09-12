"""bt.detect == ict_setup.detect_ict (canli pencere) esdegerlik testi."""
import sys, os, random, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ict_setup, bt
random.seed(3)
D = bt.D
syms = [s for s in json.load(open(os.path.join(D, "_semboller.json"))) if os.path.exists(os.path.join(D, s + ".npy"))]
LOOK = {"1h": 250, "4h": 320, "6h": 250, "12h": 250, "1d": 250}
esit = fark = pass_e = 0; farklar = []
for _ in range(600):
    sym = random.choice(syms); tf = random.choice(list(LOOK)); side = random.choice((1, -1))
    a = bt.resample(bt.yukle(sym), tf)
    if len(a) < 400: continue
    d = bt.hazirla(a, bt.P0)
    i = random.randrange(LOOK[tf], len(a))
    w = a[i - LOOK[tf] + 1:i + 1]
    df = pd.DataFrame({"open": w[:, 1], "high": w[:, 2], "low": w[:, 3], "close": w[:, 4]})
    p1, s1 = ict_setup.detect_ict(df, side)
    p2, s2 = bt.detect(d, i, side, bt.P0)
    if (p1 is None) != (p2 is None):
        fark += 1; farklar.append((sym, tf, side, i, s1, s2))
    elif p1 is None:
        esit += 1
    else:
        ok = abs(p1["entry"] - p2["entry"]) < 1e-9 and abs(p1["stop"] - p2["stop"]) < 1e-9
        esit += ok; fark += (not ok); pass_e += 1
print("esit", esit, "fark", fark, "PASS-esit", pass_e)
for f in farklar[:10]: print(" ", f)
# PASS oranini yukseltmek icin: sadece PASS olan barlarda hedefli test
n_pass = n_ok = 0
for sym in random.sample(syms, 10):
    for tf in ("1h", "4h"):
        a = bt.resample(bt.yukle(sym), tf); d = bt.hazirla(a, bt.P0)
        for i in range(LOOK[tf], len(a)):
            for side in (1, -1):
                p2, _ = bt.detect(d, i, side, bt.P0)
                if not p2: continue
                w = a[i - LOOK[tf] + 1:i + 1]
                df = pd.DataFrame({"open": w[:, 1], "high": w[:, 2], "low": w[:, 3], "close": w[:, 4]})
                p1, s1 = ict_setup.detect_ict(df, side)
                n_pass += 1
                n_ok += bool(p1) and abs(p1["entry"] - p2["entry"]) < 1e-9 and abs(p1["stop"] - p2["stop"]) < 1e-9
print("bt PASS:", n_pass, "canli da PASS ve ayni seviye:", n_ok)
