"""ICT canli mantiginin (ict_setup.detect_ict + screener.evaluate_position) sadik backtesti.
- Tespit: her kapanmis mumda, canli kodla ayni kurallar (esdegerlik testi: esdeger_test.py)
- Takip: 24s bekleyen limit, dolum mumunda stop (kotumser), ayni mumda TP sayilmaz,
  7 gun zaman asimi, sembol kilidi, 20s cooldown
- Maliyet: maker giris + taker cikis + kayma (R cinsinden, stop mesafesine bolunur)
"""
import os, sys, json
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TF_H = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}

P0 = dict(SWING_K=3, DISP_MULT=1.5, MIN_STOP_PCT=0.006, MAX_STOP_PCT=0.05, TP_R=5.0,
          LOOKBACK_BOS=40, OB_ARAMA=12, MAX_ENTRY_DIST_R=1.5, MAX_BOS_YAS=20,
          PENDING_H=24, MAX_DAYS=7, COOLDOWN_H=20,
          FEE_MAKER=0.0002, FEE_TAKER=0.0005, SLIP=0.0005)


def yukle(sym):
    return np.load(os.path.join(D, sym + ".npy"))


def resample(a, tf):
    """1h -> tf (UTC hizali, sadece tam gruplar). Kolonlar: ot,o,h,l,c,v"""
    k = TF_H[tf]
    if k == 1:
        return a
    g = (a[:, 0] // (k * 3600_000)).astype(np.int64)
    idx = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate([[0], idx]); ends = np.concatenate([idx, [len(a)]])
    out = []
    for s, e in zip(starts, ends):
        if e - s != k:
            continue
        b = a[s:e]
        out.append([g[s] * k * 3600_000, b[0, 1], b[:, 2].max(), b[:, 3].min(), b[-1, 4], b[:, 5].sum()])
    return np.array(out)


def hazirla(a, P):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    n = len(c)
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    atr = np.full(n, np.nan)
    if len(tr) > 14:
        x = tr[:14].mean(); atr[14] = x
        for i in range(15, n):
            x = (x * 13 + tr[i - 1]) / 14; atr[i] = x
    k = P["SWING_K"]
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(k, n - k):
        w = h[i - k:i + k + 1]
        if h[i] == w.max() and w.argmax() == k: sh[i] = True
        w = l[i - k:i + k + 1]
        if l[i] == w.min() and w.argmin() == k: sl[i] = True
    return dict(ot=a[:, 0], o=o, h=h, l=l, c=c, atr=atr, sh=sh, sl=sl, n=n,
                sh_idx=np.flatnonzero(sh), sl_idx=np.flatnonzero(sl))


def detect(d, i, side, P):
    """ict_setup.detect_ict'in i barinda kapanan pencere icin esdegeri. Return plan|None, sebep"""
    o, h, l, c, atr = d["o"], d["h"], d["l"], d["c"], d["atr"]
    if i < 80: return None, "veri_yetersiz"
    if np.isnan(atr[i]): return None, "atr_yok"
    idx = d["sh_idx"] if side == 1 else d["sl_idx"]
    px = h if side == 1 else l
    lim = i - P["SWING_K"]              # flags[:i-k] -> indeks < i-k
    m = np.searchsorted(idx, lim)       # idx[:m] < lim
    if m < 2: return None, "swing_yok"
    son_swing = idx[m - 1]
    swing_seviye = px[son_swing]
    bos_i = None
    for j in range(max(0, i - P["LOOKBACK_BOS"]), i):
        if j <= son_swing: continue
        if (c[j] > swing_seviye) if side == 1 else (c[j] < swing_seviye):
            bos_i = j; break
    if bos_i is None: return None, "bos_yok"
    if abs(c[bos_i] - o[bos_i]) < P["DISP_MULT"] * atr[bos_i]: return None, "displacement_zayif"
    fvg = None
    for j in range(max(bos_i - 2, 2), min(bos_i + 4, i + 1)):
        if side == 1 and l[j] > h[j - 2]: fvg = (h[j - 2], l[j]); break
        if side == -1 and h[j] < l[j - 2]: fvg = (h[j], l[j - 2]); break
    if fvg is None: return None, "fvg_yok"
    ob = None
    for j in range(bos_i, max(bos_i - P["OB_ARAMA"], 0), -1):
        if side == 1 and c[j] < o[j]: ob = (l[j], h[j]); break
        if side == -1 and c[j] > o[j]: ob = (l[j], h[j]); break
    if ob is None: return None, "ob_yok"
    if i > bos_i + 1:
        if side == 1 and l[bos_i + 1:i].min() < ob[0]: return None, "bolge_dolmus"
        if side == -1 and h[bos_i + 1:i].max() > ob[1]: return None, "bolge_dolmus"
    lo, hi = ob
    entry = (lo + hi) / 2.0
    if side == 1:
        stop = lo * 0.999
        if stop >= entry or (entry - stop) / entry < P["MIN_STOP_PCT"]: stop = entry * (1 - P["MIN_STOP_PCT"])
        risk = entry - stop
    else:
        stop = hi * 1.001
        if stop <= entry or (stop - entry) / entry < P["MIN_STOP_PCT"]: stop = entry * (1 + P["MIN_STOP_PCT"])
        risk = stop - entry
    if risk <= 0: return None, "seviye_gecersiz"
    sp = risk / entry
    if sp < P["MIN_STOP_PCT"]: return None, "stop_cok_dar"
    if sp > P["MAX_STOP_PCT"]: return None, "stop_cok_genis"
    son = c[i]
    if (i - bos_i) > P["MAX_BOS_YAS"]: return None, "bayat_kurulum"
    mes = abs(entry - son) / risk
    if mes > P["MAX_ENTRY_DIST_R"]: return None, "giris_cok_uzak"
    if side == 1 and son < entry: return None, "bolge_gecilmis"
    if side == -1 and son > entry: return None, "bolge_gecilmis"
    if side == 1 and son < stop: return None, "kacirilmis"
    if side == -1 and son > stop: return None, "kacirilmis"
    return dict(entry=entry, stop=stop, risk=risk, tp=entry + side * P["TP_R"] * risk,
                stop_pct=sp, mesafe_r=mes, bos_yas=i - bos_i, ob=ob), "PASS"


def simule(d, i, side, pl, P, tf):
    """Sinyal i barinda kapandi. Return (sonuc, r, kapanis_bar_idx)"""
    ot, h, l, c = d["ot"], d["h"], d["l"], d["c"]
    n = d["n"]; tfms = TF_H[tf] * 3600_000
    sig_close = ot[i] + tfms
    deadline = sig_close + P["PENDING_H"] * 3600_000
    e, st, tp = pl["entry"], pl["stop"], pl["tp"]
    fill = None
    for j in range(i + 1, n):
        if ot[j] > deadline: break
        if (l[j] <= e) if side == 1 else (h[j] >= e):
            fill = j; break
        if (l[j] <= st) if side == 1 else (h[j] >= st):
            return "cancelled", 0.0, j
    if fill is None:
        return "cancelled", 0.0, min(i + 1 + int(P["PENDING_H"] / TF_H[tf]), n - 1)
    tmax = sig_close + P["MAX_DAYS"] * 86400_000
    for j in range(fill, n):
        if ot[j] > tmax:
            return "timeout", side * (c[j - 1] - e) / pl["risk"], j
        if (l[j] <= st) if side == 1 else (h[j] >= st):
            return "stopped", -1.0, j
        if j > fill and ((h[j] >= tp) if side == 1 else (l[j] <= tp)):
            return "target_done", P["TP_R"], j
    return "open", None, n - 1


def ema(x, span):
    a = 2 / (span + 1); out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def gunluk_trend(a1h):
    """her gun icin (gun_ot, trend) : yukselis/dusus/kararsiz -- o gunun KAPANISINDA"""
    dd = resample(a1h, "1d")
    if len(dd) < 100: return None
    c = dd[:, 4]; e55, e99 = ema(c, 55), ema(c, 99)
    t = np.where((c >= e55) & (c >= e99), 1, np.where((c < e55) & (c < e99), -1, 0))
    return dd[:, 0], t


def trend_at(gt, t_ms):
    """t_ms aninda SON KAPANMIS gunun trendi"""
    if gt is None: return 0, False
    ots, t = gt
    k = np.searchsorted(ots, t_ms - 86400_000, side="right") - 1   # gun kapanisi <= t
    if k < 99: return 0, False
    return int(t[k]), True


def kos(sym, tf, side, P, gt_coin, gt_btc, a1h=None):
    a = resample(a1h if a1h is not None else yukle(sym), tf)
    if len(a) < 200: return []
    d = hazirla(a, P)
    out = []; kilit_bitis = -1; son_sinyal_ms = -1e18
    for i in range(80, d["n"] - 1):
        if i <= kilit_bitis: continue
        if d["ot"][i] - son_sinyal_ms < P["COOLDOWN_H"] * 3600_000: continue
        pl, sebep = detect(d, i, side, P)
        if not pl: continue
        res, r, jk = simule(d, i, side, pl, P, tf)
        if res == "open": break
        son_sinyal_ms = d["ot"][i]; kilit_bitis = jk
        sig_close = d["ot"][i] + TF_H[tf] * 3600_000
        ct, ok1 = trend_at(gt_coin, sig_close); bt, ok2 = trend_at(gt_btc, sig_close)
        maliyet = (P["FEE_MAKER"] + P["FEE_TAKER"] + P["SLIP"]) / pl["stop_pct"] if res != "cancelled" else 0.0
        out.append(dict(sym=sym, tf=tf, side=side, t=int(sig_close), res=res, r=r, net=(r - maliyet) if r is not None else None,
                        sp=round(pl["stop_pct"], 5), mes=round(pl["mesafe_r"], 3), yas=pl["bos_yas"],
                        ct=ct, bt=bt))
    return out


_GT_BTC = {}
def isle(args):
    """Process-havuzu worker'i (Windows: modul seviyesinde olmali)."""
    sym, tfs, P = args
    if "x" not in _GT_BTC:
        _GT_BTC["x"] = gunluk_trend(yukle("BTCUSDT"))
    a1h = yukle(sym); gt = gunluk_trend(a1h); res = []
    for tf in tfs:
        for side in (1, -1):
            res += kos(sym, tf, side, P, gt, _GT_BTC["x"], a1h)
    return res


if __name__ == "__main__":
    import argparse, time
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="1h,4h,6h,12h,1d")
    ap.add_argument("--out", default="trades_base.json")
    ap.add_argument("--set", default="", help="P override: KEY=val,KEY=val")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    P = dict(P0)
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("="); P[k] = float(v)
    syms = json.load(open(os.path.join(D, "_semboller.json")))
    syms = [s for s in syms if os.path.exists(os.path.join(D, s + ".npy"))]
    if a.n: syms = syms[:a.n]
    tfs = a.tfs.split(",")
    t0 = time.time(); allt = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(isle, [(s, tfs, P) for s in syms]), 1):
            allt += r
            if i % 20 == 0: print(i, len(syms), len(allt), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(P=P, trades=allt), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out), "w"))
    print("islem:", len(allt), "sure", round(time.time() - t0), "s")
