"""Top-100 + BTC icin 1d mum verisi (1500 gun) -> data1d/<SYM>.npy"""
import sys, os, time, json
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data1d"); os.makedirs(D, exist_ok=True)
syms = json.load(open(os.path.join(os.path.dirname(D), "data", "_semboller.json")))
since0 = int(time.time() * 1000) - 1500 * 86_400_000

def indir(sym):
    out = os.path.join(D, sym + ".npy")
    if os.path.exists(out): return sym, "var"
    since, rows = since0, []
    while True:
        for _ in range(5):
            try:
                r = requests.get("https://data-api.binance.vision/api/v3/klines",
                                 params={"symbol": sym, "interval": "1d", "startTime": since, "limit": 1000}, timeout=30)
                if r.status_code == 200: break
            except Exception: pass
            time.sleep(3)
        else: return sym, "HATA"
        b = r.json()
        if not b: break
        rows += [[x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])] for x in b]  # qav = USDT hacmi
        since = b[-1][0] + 86_400_000
        if len(b) < 1000: break
    if len(rows) < 120: return sym, f"kisa {len(rows)}"
    np.save(out, np.array(rows[:-1], dtype=np.float64))   # son (acik) gunu at
    return sym, len(rows)

with ThreadPoolExecutor(8) as ex:
    for s, r in ex.map(indir, syms): print(s, r, flush=True)
print("bitti")
