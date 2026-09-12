"""Top-100 vadeli coin icin 1h mum verisi (730 gun) -> data/<SYM>.npy (ot,o,h,l,c,v)"""
import sys, os, time, json, types
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
for m in ['matplotlib', 'mplfinance']:
    x = types.ModuleType(m); x.use = lambda *a, **k: None; sys.modules[m] = x
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import screener

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(D, exist_ok=True)
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 730

def indir(sym):
    out = os.path.join(D, sym + ".npy")
    if os.path.exists(out):
        return sym, "var"
    since = int(time.time() * 1000) - DAYS * 86_400_000
    rows = []
    while True:
        for _ in range(5):
            try:
                r = requests.get(f"{screener.BASE_URL}/api/v3/klines",
                                 params={"symbol": sym, "interval": "1h", "startTime": since, "limit": 1000}, timeout=30)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        else:
            return sym, "HATA"
        b = r.json()
        if not b:
            break
        rows += [[x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in b]
        since = b[-1][0] + 3600_000
        if len(b) < 1000:
            break
    if len(rows) < 500:
        return sym, f"kisa {len(rows)}"
    np.save(out, np.array(rows, dtype=np.float64))
    return sym, len(rows)

syms = screener.get_usdt_symbols(tum=True)[:screener.TOP_N]
if "BTCUSDT" not in syms:
    syms.append("BTCUSDT")
json.dump(syms, open(os.path.join(D, "_semboller.json"), "w"))
print(len(syms), "sembol")
with ThreadPoolExecutor(8) as ex:
    for i, (s, r) in enumerate(ex.map(indir, syms), 1):
        print(i, s, r, flush=True)
print("bitti")
