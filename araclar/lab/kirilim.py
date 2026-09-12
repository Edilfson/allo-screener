"""AILE: kirilim -- intraday trend/kirilim stratejileri (1h verisi, 1h ve 4h dilim).

Tum veri/simulasyon harness.py uzerinden. rapor(..., kaynak="1h") kullanilir
(TRAIN < 2025-07-01, VALID 2025-07..12).

Hipotezler:
  H1 donchian : N-bar en yuksek KAPANIS ustunde kapanis -> LONG (piyasa, t+1 acilis)
                stop = giris - k*ATR14 ; cikis = M-bar en dusuk kapanis altinda kapanis
                (cikis_i) veya ATR iz suren stop. SHORT simetrik.
  H2 volb     : c[i] > c[i-1] + f*ATR[i-1] ve v[i] > 1.5 * 20-bar ort hacim -> LONG.
  H3 overlay  : BTC gunluk MA50 rejimi (harness.rejim_at) / sadece-long / sadece-short.
  H4 cikis    : trend cikisi (TP yok) vs sabit 3R / 5R TP.

Ileriye bakma yok: tum gostergeler i barinin KAPANISINDA bilinen veriden; giris t+1 acilisi;
cikis_i sadece o barin kapanisinda tetiklenen kosuldan (ileri tarama, gecmis bilgiyle).

Kullanim:
  PYTHONIOENCODING=utf-8 python kirilim.py --faz a           # donchian grid (TRAIN)
  PYTHONIOENCODING=utf-8 python kirilim.py --faz b|c|d       # cikis/overlay/volb (TRAIN)
  PYTHONIOENCODING=utf-8 python kirilim.py --faz valid --ad X --ad Y --ad Z   # en fazla 3
  ... --n 10   ilk 10 coin ile hizli deneme
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

AILE = "kirilim"
CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuc_kirilim.json")


# ------------------------------------------------------------------ gostergeler
def _swmax(x, N):
    """out[i] = max(x[i-N:i])  (i HARIC, tamamen gecmis)"""
    out = np.full(len(x), np.nan)
    if len(x) > N:
        w = np.lib.stride_tricks.sliding_window_view(x, N)
        out[N:] = w.max(axis=1)[:len(x) - N]
    return out


def _swmean(x, N):
    """out[i] = mean(x[i-N:i])  (i HARIC)"""
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    if len(x) > N:
        out[N:] = (cs[N:-1] - cs[:-N - 1]) / N
    return out


def _atr(a, N=14):
    """out[i] = mean(TR[i-N+1..i]) -- i barinin kapanisinda bilinir"""
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(c), np.nan)
    cs = np.cumsum(tr)
    if len(c) >= N:
        out[N - 1:] = (cs[N - 1:] - np.concatenate([[0.0], cs[:-N]])) / N
    return out


def _next_true(cond):
    """out[i] = cond'un True oldugu ilk j>=i (yoksa n-1)"""
    n = len(cond)
    out = np.empty(n, np.int64)
    nx = n - 1
    for i in range(n - 1, -1, -1):
        if cond[i]:
            nx = i
        out[i] = nx
    return out


def _trail_cikis(hi, lo, c, atr, i, side, ktrail, nmax=600):
    """Giristen itibaren ATR iz suren stop: ekstrem -/+ ktrail*ATR kapanisla delinirse cikis bari."""
    n = len(c)
    ext = c[i]
    son = min(n - 1, i + nmax)
    for j in range(i + 1, son + 1):
        if side == 1:
            if hi[j] > ext:
                ext = hi[j]
            if c[j] < ext - ktrail * atr[j]:
                return j
        else:
            if lo[j] < ext:
                ext = lo[j]
            if c[j] > ext + ktrail * atr[j]:
                return j
    return son


# ------------------------------------------------------------------ sinyal uretimi
def _cache(cc, k, f):
    if k not in cc:
        cc[k] = f()
    return cc[k]


def _uret(a, cfg, cc):
    c, hi, lo, v, ot = a[:, 4], a[:, 2], a[:, 3], a[:, 5], a[:, 0]
    n = len(c)
    tfms = H.TF_H[cfg["tf"]] * 3600_000
    atr = _cache(cc, "atr14", lambda: _atr(a, 14))
    M = cfg["M"]
    k = cfg["k"]
    tip = cfg["tip"]

    if tip == "donchian":
        N = cfg["N"]
        mx = _cache(cc, f"cmax{N}", lambda: _swmax(c, N))
        mn = _cache(cc, f"cmin{N}", lambda: -_swmax(-c, N))
        up = c > mx
        dn = c < mn
        upl = up & ~np.concatenate([[False], up[:-1]])       # taze kirilim
        dnl = dn & ~np.concatenate([[False], dn[:-1]])
        warm = N
    else:  # volb
        f = cfg["f"]
        pc = np.concatenate([[np.nan], c[:-1]])
        pa = np.concatenate([[np.nan], atr[:-1]])
        vma = _cache(cc, "vma20", lambda: _swmean(v, 20))
        volok = v > cfg["vmult"] * vma
        upl = (c > pc + f * pa) & volok
        dnl = (c < pc - f * pa) & volok
        upl = np.nan_to_num(upl, nan=0).astype(bool)
        dnl = np.nan_to_num(dnl, nan=0).astype(bool)
        warm = 25

    xM = _cache(cc, f"cmax{M}", lambda: _swmax(c, M))
    nM = _cache(cc, f"cmin{M}", lambda: -_swmax(-c, M))
    nxtL = _cache(cc, f"nxL{M}", lambda: _next_true(np.nan_to_num(c < nM, nan=0).astype(bool)))
    nxtS = _cache(cc, f"nxS{M}", lambda: _next_true(np.nan_to_num(c > xM, nan=0).astype(bool)))

    yon = cfg.get("yon", "both")
    rej = cfg.get("rejim", 0)
    tpr = cfg.get("tp")
    cik = cfg.get("cikis", "kanal")
    ktr = cfg.get("ktrail", 3.0)
    start = max(warm, M, 20) + 2
    sig = []
    for i in range(start, n - 1):
        ai = atr[i]
        if not np.isfinite(ai) or ai <= 0:
            continue
        for side, trig, nxt in ((1, upl, nxtL), (-1, dnl, nxtS)):
            if not trig[i]:
                continue
            if (yon == "long" and side < 0) or (yon == "short" and side > 0):
                continue
            if rej and H.rejim_at(int(ot[i]) + tfms, 50) != side:
                continue
            stop = c[i] - side * k * ai
            if cik == "atr":
                ci = _trail_cikis(hi, lo, c, atr, i, side, ktr)
            else:
                ci = int(nxt[i + 1])
            d = dict(i=i, side=side, entry=None, stop=float(stop), tp=None,
                     cikis_i=int(ci), max_days=400)
            if tpr:
                d["tp"] = float(c[i] + side * tpr * abs(c[i] - stop))
            sig.append(d)
    return sig


# ------------------------------------------------------------------ kosucu (modul seviyesi -> ProcessPool)
def _sym_kos(arg):
    sym, tf, cfgs = arg
    try:
        a = H.veri_1h(sym)
    except Exception:
        return sym, {}
    if tf != "1h":
        a = H.resample(a, tf)
    if len(a) < 300:
        return sym, {}
    cc = {}
    out = {}
    for cfg in cfgs:
        try:
            sig = _uret(a, cfg, cc)
            out[cfg["ad"]] = H.sim_islem(a, sig, tf=tf, cooldown_h=cfg.get("cd", 20))
        except Exception as e:
            out[cfg["ad"]] = []
    return sym, out


def kos(cfgs, syms, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    grup = {}
    for cfg in cfgs:
        grup.setdefault(cfg["tf"], []).append(cfg)
    tum = {cfg["ad"]: [] for cfg in cfgs}
    isler = [(s, tf, cl) for tf, cl in grup.items() for s in syms]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for sym, res in ex.map(_sym_kos, isler, chunksize=1):
            for ad, tr in res.items():
                tum[ad] += tr
    print(f"  [{len(isler)} is, {time.time()-t0:.0f}s]", flush=True)
    return tum


# ------------------------------------------------------------------ konfigurasyonlar
def _ad(cfg):
    p = [cfg["tip"], cfg["tf"]]
    if cfg["tip"] == "donchian":
        p.append(f"N{cfg['N']}")
    else:
        p.append(f"f{cfg['f']}v{cfg['vmult']}")
    p.append(f"M{cfg['M']}")
    p.append(f"k{cfg['k']}")
    if cfg.get("cikis", "kanal") != "kanal":
        p.append(f"trail{cfg.get('ktrail')}")
    if cfg.get("tp"):
        p.append(f"tp{cfg['tp']}R")
    if cfg.get("yon", "both") != "both":
        p.append(cfg["yon"])
    if cfg.get("rejim"):
        p.append("rej")
    return "_".join(str(x) for x in p)


def _mk(**kw):
    kw.setdefault("tip", "donchian")
    if kw["tip"] == "donchian" and "M" not in kw:
        kw["M"] = max(10, kw["N"] // 2)
    kw["ad"] = _ad(kw)
    return kw


def faz_a():
    out = []
    for tf in ("4h", "1h"):
        for N in (20, 40, 55, 80):
            for k in (1.5, 2.0, 3.0):
                out.append(_mk(tf=tf, N=N, k=k))
    return out


def faz_b(base):
    """cikis / TP varyantlari: base = [(tf,N,k), ...]"""
    out = []
    for tf, N, k in base:
        M = max(10, N // 2)
        for kt in (2.0, 3.0):
            out.append(_mk(tf=tf, N=N, k=k, M=M, cikis="atr", ktrail=kt))
        for tp in (3, 5):
            out.append(_mk(tf=tf, N=N, k=k, M=M, tp=tp))
        for Mv in (max(5, N // 4), N):
            if Mv != M:
                out.append(_mk(tf=tf, N=N, k=k, M=Mv))
    return out


def faz_c(base):
    out = []
    for tf, N, k in base:
        M = max(10, N // 2)
        for yon in ("long", "short"):
            out.append(_mk(tf=tf, N=N, k=k, M=M, yon=yon))
        out.append(_mk(tf=tf, N=N, k=k, M=M, rejim=1))
        out.append(_mk(tf=tf, N=N, k=k, M=M, rejim=1, yon="long"))
    return out


def faz_d():
    out = []
    for tf in ("4h", "1h"):
        for f in (0.5, 1.0, 1.5):
            for k in (2.0, 3.0):
                out.append(_mk(tip="volb", tf=tf, f=f, vmult=1.5, k=k, M=20))
    return out


def faz_e():
    """Faz C kazananlarinin komsu + kombinasyon taramasi (TRAIN)."""
    out = []
    # 1h short kirilimi: N, M, k komsulari
    for N in (40, 55, 80):
        for k in (1.5, 2.0, 3.0):
            out.append(_mk(tf="1h", N=N, k=k, yon="short"))
    for M in (13, 40, 55):
        out.append(_mk(tf="1h", N=55, k=2.0, M=M, yon="short"))
    out.append(_mk(tf="1h", N=55, k=2.0, yon="short", tp=5))
    out.append(_mk(tf="1h", N=55, k=2.0, yon="short", tp=3))
    out.append(_mk(tf="1h", N=55, k=2.0, yon="short", cikis="atr", ktrail=3.0))
    out.append(_mk(tf="1h", N=55, k=2.0, yon="short", rejim=1))
    # 1h long + rejim komsulari
    for N in (40, 55, 80):
        for k in (1.5, 2.0, 3.0):
            out.append(_mk(tf="1h", N=N, k=k, yon="long", rejim=1))
    out.append(_mk(tf="1h", N=55, k=2.0, yon="long", rejim=1, tp=5))
    # 4h long + rejim komsulari
    for N in (20, 30, 40):
        for k in (1.5, 2.0, 3.0):
            out.append(_mk(tf="4h", N=N, k=k, yon="long", rejim=1))
    out.append(_mk(tf="4h", N=20, k=2.0, M=5, yon="long", rejim=1))
    out.append(_mk(tf="4h", N=20, k=2.0, M=20, yon="long", rejim=1))
    out.append(_mk(tf="4h", N=20, k=2.0, yon="long", rejim=1, tp=3))
    # 4h short
    for k in (1.5, 2.0, 3.0):
        out.append(_mk(tf="4h", N=20, k=k, yon="short"))
    return out


def cfg_bul(ad, base):
    for f in (faz_a(), faz_b(base), faz_c(base), faz_d(), faz_e()):
        for c in f:
            if c["ad"] == ad:
                return c
    return None


# ------------------------------------------------------------------ main
def _yaz(faz, cfgs, tum, bolumler):
    sat = []
    for cfg in cfgs:
        tr = tum[cfg["ad"]]
        satir = {"ad": cfg["ad"], "params": {k: v for k, v in cfg.items() if k != "ad"}}
        for b in bolumler:
            r = H.rapor(tr, b, kaynak="1h")
            satir[b] = r
            H.kaydet(AILE, cfg["ad"], satir["params"], b, r, not_=f"faz={faz}")
        sat.append(satir)
    sat.sort(key=lambda s: -(s[bolumler[0]].get("ort_net") or -9))
    for s in sat:
        r = s[bolumler[0]]
        print(f"{s['ad']:44s} n={r.get('n',0):5d} net={r.get('ort_net')} "
              f"ga99={r.get('ga99')} win={r.get('win')} y1={r.get('yari1')} y2={r.get('yari2')}",
              flush=True)
    return sat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="a")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--ad", action="append", default=[])
    ap.add_argument("--base", default="4h:55:2.0,1h:55:2.0")
    a = ap.parse_args()

    syms = H.semboller_1h()
    if a.n:
        syms = syms[:a.n]
    base = []
    for x in a.base.split(","):
        tf, N, k = x.split(":")
        base.append((tf, int(N), float(k)))

    if a.faz == "a":
        cfgs = faz_a()
    elif a.faz == "b":
        cfgs = faz_b(base)
    elif a.faz == "c":
        cfgs = faz_c(base)
    elif a.faz == "d":
        cfgs = faz_d()
    elif a.faz == "e":
        cfgs = faz_e()
    elif a.faz == "valid":
        cfgs = [cfg_bul(x, base) for x in a.ad]
        assert all(cfgs) and len(cfgs) <= 3, "en fazla 3 ve gecerli ad"
    else:
        raise SystemExit("faz?")

    print(f"faz={a.faz} cfg={len(cfgs)} sym={len(syms)}", flush=True)
    tum = kos(cfgs, syms, a.w)
    bol = ["train", "valid"] if a.faz == "valid" else ["train"]
    _yaz(a.faz, cfgs, tum, bol)


if __name__ == "__main__":
    main()
