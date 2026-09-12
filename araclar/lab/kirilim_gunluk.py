"""AILE: kirilim_gunluk - Donchian / kirilim stratejileri GUNLUK (1d) veride, 2017'den beri.

Kaynak: harness.veri_1d (data1d_all). Bolum: TRAIN < 2025-01-01, VALID 2025 (kaynak="1d").
Evren filtresi: sinyal aninda coinin son 30 gunluk ortalama USDT hacmi (kolon 5) >= 5M USDT.
Ileriye bakma yok: tum gostergeler t barini ve oncesini kullanir; sinyal t kapanisinda,
piyasa girisi t+1 acilisinda (sim_islem), limit girisi t+1..t+pending araliginda.

Kullanim:
  PYTHONIOENCODING=utf-8 python kirilim_gunluk.py --varyant h1        # Donchian taramasi (TRAIN)
  ... --varyant h2      # limit geri cekilme girisi
  ... --varyant h3      # rejim overlay
  ... --varyant komsu   # secilen adayin komsulari
  ... --varyant valid   # en fazla 3 aday, TEK sefer
  ... --varyant h4      # portfoy dogrulamasi (sim_portfoy)
"""
import os, sys, json, argparse, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

AILE = "kirilim_gunluk"
MIN_VOL = 5e6          # 30g ort USDT hacim esigi
MIN_BAR = 200          # coin icin minimum gun sayisi
MAX_DAYS = 365         # sim_islem zaman asimi tavani (gun)


# ---------------------------------------------------------------- gostergeler (hepsi nedensel)
def _roll_max(x, N):
    """out[i] = max(x[i-N .. i-1]) ; ilk N eleman nan"""
    n = len(x); out = np.full(n, np.nan)
    if n > N:
        sw = np.lib.stride_tricks.sliding_window_view(x, N)   # sw[j] = x[j..j+N-1]
        out[N:] = sw[:n - N].max(axis=1)
    return out


def _roll_min(x, N):
    n = len(x); out = np.full(n, np.nan)
    if n > N:
        sw = np.lib.stride_tricks.sliding_window_view(x, N)
        out[N:] = sw[:n - N].min(axis=1)
    return out


def _roll_mean(x, N):
    """out[i] = mean(x[i-N+1 .. i])  (i dahil, nedensel)"""
    n = len(x); out = np.full(n, np.nan)
    if n >= N:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        out[N - 1:] = (cs[N:] - cs[:-N]) / N
    return out


def ozellikler(a):
    o, h, l, c, v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return dict(ot=a[:, 0], o=o, h=h, l=l, c=c,
                atr=_roll_mean(tr, 14), ma100=_roll_mean(c, 100), vol30=_roll_mean(v, 30))


def _btc_rejim_ic(ts, N):
    """rejim_at(t_ms, N) ile birebir ayni; vektorlestirilmis."""
    ots, r, _ = H.btc_ma_rejim(N)
    k = np.searchsorted(ots, ts - 86400_000, side="right") - 1
    k = np.clip(k, 0, len(r) - 1)
    return np.where(k >= N, r[k], 0)


# ---------------------------------------------------------------- sinyal uretimi
def sinyaller(F, cfg):
    """cfg: N,M,k,yon,tp,cikis(kanal|atr),ktrail,giris(piyasa|limit),limit_pct,pending_h,rejim,ma100"""
    c, hh_ot = F["c"], F["ot"]
    n = len(c)
    N, M, kk, yon = cfg["N"], cfg["M"], cfg["k"], cfg["yon"]
    side = 1 if yon == "long" else -1
    atr, ma100, vol30 = F["atr"], F["ma100"], F["vol30"]

    hi = _roll_max(c, N); lo = _roll_min(c, N)
    ex_lo = _roll_min(c, M); ex_hi = _roll_max(c, M)

    if side == 1:
        trig = c > hi
        exit_cond = c < ex_lo
    else:
        trig = c < lo
        exit_cond = c > ex_hi
    trig &= np.isfinite(atr) & (atr > 0) & np.isfinite(vol30) & (vol30 >= MIN_VOL)
    if cfg.get("ma100"):
        trig &= np.isfinite(ma100) & ((c > ma100) if side == 1 else (c < ma100))
    if cfg.get("rejim"):
        rj = _btc_rejim_ic(hh_ot + 86400_000, cfg["rejim"])
        trig &= (rj == side)
    warm = max(N, M, 100 if cfg.get("ma100") else 0, 30, 14) + 1
    trig[:warm] = False
    trig[n - 2:] = False

    # ---- ESLESMIS PLASEBO: ayni sayida giris, ayni stop/TP/cikis mekanigi, RASTGELE bar
    if cfg.get("plasebo") is not None:
        uygun = (np.isfinite(atr) & (atr > 0) & np.isfinite(vol30) & (vol30 >= MIN_VOL))
        if cfg.get("plasebo_overlay"):     # overlay'i koru, sadece Donchian tetigini rastgelelestir
            if cfg.get("ma100"):
                uygun &= np.isfinite(ma100) & ((c > ma100) if side == 1 else (c < ma100))
            if cfg.get("rejim"):
                uygun &= (_btc_rejim_ic(hh_ot + 86400_000, cfg["rejim"]) == side)
        uygun[:warm] = False; uygun[n - 2:] = False
        ok = np.flatnonzero(uygun); kac = int(trig.sum())
        trig = np.zeros(n, bool)
        if kac and len(ok):
            rng = np.random.default_rng(int(cfg["plasebo"]) * 1000003 + (int(hh_ot[0]) % 100000))
            trig[rng.choice(ok, size=min(kac, len(ok)), replace=False)] = True

    # kanal cikisi icin: her j'den sonraki ilk cikis bari
    nxt = np.full(n, -1, dtype=np.int64)
    nx = -1
    for j in range(n - 1, -1, -1):
        if exit_cond[j]:
            nx = j
        nxt[j] = nx

    cik_tip = cfg.get("cikis", "kanal")
    ktrail = cfg.get("ktrail", 3.0)
    gir = cfg.get("giris", "piyasa")
    lpct = cfg.get("limit_pct", 0.0)
    pend = cfg.get("pending_h", 96)
    tpR = cfg.get("tp")

    out = []
    for i in np.flatnonzero(trig):
        i = int(i)
        if gir == "piyasa":
            ref = c[i]; entry = None
        else:
            ref = c[i] * (1 - side * lpct); entry = float(ref)
        risk = kk * atr[i]
        stop = ref - side * risk
        if stop <= 0:
            continue
        tp = float(ref + side * tpR * risk) if tpR else None
        # cikis bar indeksi (fill = i+1 varsayimi; sim_islem j > fill sarti ariyor)
        if cik_tip == "kanal":
            j0 = i + 2
            ci = int(nxt[j0]) if j0 < n and nxt[j0] >= 0 else None
        else:  # atr iz suren: giristen sonra en iyi kapanistan ktrail*ATR geri cekilme
            ci = None
            best = c[i + 1] if i + 1 < n else c[i]
            for j in range(i + 2, n):
                best = max(best, c[j]) if side == 1 else min(best, c[j])
                if (c[j] <= best - ktrail * atr[j]) if side == 1 else (c[j] >= best + ktrail * atr[j]):
                    ci = j; break
        s = dict(i=i, side=side, entry=entry, stop=float(stop), tp=tp,
                 cikis_i=ci, max_days=MAX_DAYS)
        if entry is not None:
            s["pending_h"] = pend
        out.append(s)
    return out


# ---------------------------------------------------------------- calistirma
_CACHE = {}


def _feat(sym):
    if sym not in _CACHE:
        try:
            a = H.veri_1d(sym)
        except Exception:
            _CACHE[sym] = None; return None
        if len(a) < MIN_BAR or not np.isfinite(a[:, 1:5]).all():
            _CACHE[sym] = None; return None
        _CACHE[sym] = ozellikler(a), a
    return _CACHE[sym]


def _isle(arg):
    sym, cfgs = arg
    r = _feat(sym)
    if r is None:
        return {}
    F, a = r
    out = {}
    for ad, cfg in cfgs:
        sig = sinyaller(F, cfg)
        tr = H.sim_islem(a, sig, tf="1d") if sig else []
        for t in tr:
            t["sym"] = sym
        out[ad] = tr
    return out


def kos(cfgs, syms=None, isci=8):
    syms = syms or H.semboller_1d()
    res = {ad: [] for ad, _ in cfgs}
    args = [(s, cfgs) for s in syms]
    if isci > 1:
        with ProcessPoolExecutor(max_workers=isci) as ex:
            for d in ex.map(_isle, args, chunksize=8):
                for ad, tr in d.items():
                    res[ad] += tr
    else:
        for x in args:
            for ad, tr in _isle(x).items():
                res[ad] += tr
    for ad in res:
        res[ad].sort(key=lambda t: t["t"])
    return res


def ad_of(cfg):
    p = [f"N{cfg['N']}", f"M{cfg['M']}", f"k{cfg['k']}", cfg["yon"]]
    if cfg.get("tp"): p.append(f"tp{cfg['tp']}R")
    if cfg.get("cikis") == "atr": p.append(f"trail{cfg.get('ktrail')}")
    if cfg.get("giris") == "limit": p.append(f"lim{int(cfg['limit_pct']*100)}p{cfg['pending_h']}")
    if cfg.get("rejim"): p.append(f"btc{cfg['rejim']}")
    if cfg.get("ma100"): p.append("ma100")
    return "d_" + "_".join(str(x) for x in p)


def rapor_yaz(ad, cfg, tr, bolum="train", not_=""):
    rp = H.rapor(tr, bolum, "1d")
    H.kaydet(AILE, ad, cfg, bolum, rp, not_)
    return rp


def yazdir(ad, rp):
    if rp.get("ort_net") is None:
        print(f"  {ad:52s} n={rp.get('n')} (yetersiz)"); return
    yl = rp["yillar"]; poz = sum(1 for y, v in yl.items() if v[1] > 0)
    print(f"  {ad:52s} n={rp['n']:5d} brut={rp['ort_brut']:+.4f} net={rp['ort_net']:+.4f} "
          f"ga99=[{rp['ga99'][0]:+.4f},{rp['ga99'][1]:+.4f}] win={rp['win']:.3f} "
          f"y1={rp['yari1']:+.3f} y2={rp['yari2']:+.3f} yil+={poz}/{len(yl)}")


# ---------------------------------------------------------------- varyantlar
def v_h1(syms, isci):
    cfgs = []
    for N in (20, 40, 55, 100):
        for k in (2.0, 3.0):
            for yon in ("long", "short"):
                for tp in (None, 5):
                    cfgs.append(dict(N=N, M=max(2, N // 2), k=k, yon=yon, tp=tp, cikis="kanal"))
    return [(ad_of(c), c) for c in cfgs]


def v_h1b(syms, isci):
    """ATR iz suren cikis karsilastirmasi"""
    cfgs = []
    for N in (40, 55, 100):
        for yon in ("long", "short"):
            for kt in (3.0, 5.0):
                cfgs.append(dict(N=N, M=max(2, N // 2), k=2.0, yon=yon, tp=None, cikis="atr", ktrail=kt))
    return [(ad_of(c), c) for c in cfgs]


def v_h2(syms, isci):
    cfgs = []
    for N in (40, 55, 100):
        for yon in ("long", "short"):
            for pct in (0.02, 0.03):
                for ph in (72, 120):
                    cfgs.append(dict(N=N, M=max(2, N // 2), k=2.0, yon=yon, tp=None, cikis="kanal",
                                     giris="limit", limit_pct=pct, pending_h=ph))
    return [(ad_of(c), c) for c in cfgs]


def v_h3(syms, isci):
    cfgs = []
    for N in (40, 55, 100):
        for yon in ("long", "short"):
            for rj in (50, 100):
                cfgs.append(dict(N=N, M=max(2, N // 2), k=2.0, yon=yon, tp=None, cikis="kanal", rejim=rj))
            cfgs.append(dict(N=N, M=max(2, N // 2), k=2.0, yon=yon, tp=None, cikis="kanal", ma100=True))
            cfgs.append(dict(N=N, M=max(2, N // 2), k=2.0, yon=yon, tp=None, cikis="kanal", rejim=50, ma100=True))
    return [(ad_of(c), c) for c in cfgs]


VARYANT = dict(h1=v_h1, h1b=v_h1b, h2=v_h2, h3=v_h3)


# ---------------------------------------------------------------- H4: portfoy dogrulamasi
def _vol30_panel(V):
    n, m = V.shape
    X = np.nan_to_num(V)
    cs = np.vstack([np.zeros(m), np.cumsum(X, axis=0)])
    out = np.full((n, m), np.nan)
    out[29:] = (cs[30:] - cs[:-30]) / 30.0
    return out


def portfoy(cfg, syms=None, max_poz=20, isci=8, ad=None, risk=None, poz_cap=0.15):
    """En iyi islem-bazli kurali gunluk agirlik matrisine cevirip sim_portfoy ile kosar.
    Agirlik 1/max_poz; ayni anda en fazla max_poz pozisyon (once gelen alir).
    Kiyas: BTC al-tut ve evren (30g hacim >= 5M) esit-agirlik al-tut."""
    ad = ad or ad_of(cfg)
    syms = syms or H.semboller_1d()
    psyms, gun, O, C, V = H.gunluk_panel(syms, min_gun=MIN_BAR)
    gi = {int(g): i for i, g in enumerate(gun)}
    jx = {s: j for j, s in enumerate(psyms)}
    res = kos([(ad, cfg)], syms, isci)[ad]

    # (giris_gun, cikis_gun, sembol) araliklarina cevir
    grup = {}
    for t in res:
        grup.setdefault(t["sym"], []).append(t)
    araliklar = []
    for s, tl in grup.items():
        r = _feat(s)
        if r is None or s not in jx:
            continue
        ot = r[0]["ot"]
        for t in tl:
            if t["res"] == "cancelled":
                continue
            f = int(np.searchsorted(ot, t["t"]))
            if f >= len(ot) or int(ot[f]) != int(t["t"]):
                continue
            e = min(f + int(t["bar"]), len(ot) - 1)
            g0, g1 = gi.get(int(ot[f])), gi.get(int(ot[e]))
            if g0 is None or g1 is None:
                continue
            araliklar.append((g0, g1, jx[s], float(t["sp"])))
    araliklar.sort()

    n, m = C.shape
    W = np.zeros((n, m)); aktif = []   # (bitis_gun, j)
    for g0, g1, j, sp in araliklar:
        aktif = [a for a in aktif if a[0] >= g0]
        if len(aktif) >= max_poz:
            continue
        aktif.append((g1, j))
        w = 1.0 / max_poz if risk is None else min(poz_cap, risk / max(sp, 1e-6))
        W[max(0, g0 - 1):max(0, g1), j] = w
    if risk is not None:                       # toplam kaldirac <= 1
        tot = W.sum(axis=1, keepdims=True)
        W = W / np.maximum(tot, 1.0)

    eq, g = H.sim_portfoy(O, C, W)
    # kiyaslar
    Wb = np.zeros((n, m)); Wb[:, jx["BTCUSDT"]] = 1.0
    eqb, gb = H.sim_portfoy(O, C, Wb)
    v30 = _vol30_panel(V)
    lik = np.isfinite(v30) & (v30 >= MIN_VOL) & np.isfinite(C)
    cnt = lik.sum(axis=1, keepdims=True).clip(1)
    We = lik / cnt
    eqe, ge = H.sim_portfoy(O, C, We)
    return dict(gun=gun, strateji=(eq, g), btc=(eqb, gb), evren=(eqe, ge), pozisyon=int(W.astype(bool).sum(axis=1).max()),
                ort_pozisyon=round(float(W.astype(bool).sum(axis=1).mean()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--varyant", default="h1")
    ap.add_argument("--bolum", default="train")
    ap.add_argument("--coin", type=int, default=0)
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    syms = H.semboller_1d()
    if a.coin:
        syms = syms[:a.coin]
    if a.varyant in VARYANT:
        cfgs = VARYANT[a.varyant](syms, a.isci)
    else:
        cfgs = [(ad_of(c), c) for c in json.loads(a.json)]
    print(f"[{a.varyant}] {len(cfgs)} konfig x {len(syms)} coin, bolum={a.bolum}")
    res = kos(cfgs, syms, a.isci)
    for ad, cfg in cfgs:
        rp = rapor_yaz(ad, cfg, res[ad], a.bolum, not_=f"varyant={a.varyant}")
        yazdir(ad, rp)


if __name__ == "__main__":
    main()
