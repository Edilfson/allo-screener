"""positions.json sinyallerini Binance mumlarindan bagimsiz yeniden oynatir."""
import json, sys, time, os
from datetime import datetime, timezone, timedelta
import requests

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(os.path.dirname(__file__), "replay.json")
IVMS = {"1h": 3600e3, "2h": 7200e3, "4h": 14400e3, "6h": 21600e3, "12h": 43200e3, "1d": 86400e3}
PENDING_H, MAX_DAYS = 24, 7
NOW = datetime.now(timezone.utc)

def ts(s): return datetime.fromisoformat(s)
def ms(d): return int(d.timestamp() * 1000)

def klines(sym, iv, start_ms):
    out = []
    while True:
        for k in range(4):
            r = requests.get("https://data-api.binance.vision/api/v3/klines",
                             params={"symbol": sym, "interval": iv, "startTime": start_ms, "limit": 1000}, timeout=30)
            if r.status_code == 200: break
            time.sleep(2)
        else:
            return None
        raw = r.json()
        out += raw
        if len(raw) < 1000: return out
        start_ms = raw[-1][0] + 1

def replay(p, K):
    side, e, st, risk = p["side"], p["entry"], p["stop"], p["risk"]
    tp = p["tps"][0]["p"]
    sig = ts(p["signal_bar_close"]); opened = ts(p["opened_at"])
    bars = [(k[0], float(k[2]), float(k[3]), float(k[4]), k[6]) for k in K if k[6] > ms(sig)]
    deadline = ms(sig + timedelta(hours=PENDING_H))
    fill_i = None
    for i, (ot, h, l, c, ct) in enumerate(bars):
        if ot > deadline: break
        if (l <= e) if side == 1 else (h >= e):
            fill_i = i; break
    if fill_i is None:
        if bars and (bars[-1][0] > deadline or ms(NOW) > deadline):
            return {"res": "cancelled", "r": 0.0}
        return {"res": "pending", "r": None}
    tmax = ms(opened + timedelta(days=MAX_DAYS))
    same_bar_tp = False
    for i in range(fill_i, len(bars)):
        ot, h, l, c, ct = bars[i]
        if ot > tmax:
            pc = bars[i - 1][3]
            return {"res": "timeout", "r": side * (pc - e) / risk, "fill_ct": bars[fill_i][4]}
        if (l <= st) if side == 1 else (h >= st):
            return {"res": "stopped", "r": -1.0, "fill_ct": bars[fill_i][4],
                    "same_bar": i == fill_i}
        if (h >= tp) if side == 1 else (l <= tp):
            if i == fill_i:          # dolum mumunda TP: sira belirsiz -> muhafazakar: sayma
                same_bar_tp = True; continue
            return {"res": "target_done", "r": 5.0, "fill_ct": bars[fill_i][4]}
    if ms(NOW) > tmax and bars:
        return {"res": "timeout", "r": side * (bars[-1][3] - e) / risk, "fill_ct": bars[fill_i][4]}
    return {"res": "open", "r": None, "unreal": side * (bars[-1][3] - e) / risk if bars else 0}

P = json.load(open(os.path.join(REPO, "positions.json"), encoding="utf-8"))
res = []
for n, p in enumerate(P):
    K = klines(p["symbol"], p["interval"], ms(ts(p["signal_bar_close"])) - 1)
    rr = replay(p, K) if K is not None else {"res": "veri_yok", "r": None}
    res.append(rr)
    if n % 25 == 0: print(n, file=sys.stderr)
    time.sleep(0.05)
json.dump(res, open(OUT, "w"), indent=0)
print("ok", len(res))
