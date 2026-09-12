"""AILE: rejim  -- volatilite / rejim tabanli stratejiler + KONTROL (mean-reversion).

Calistirma:
    PYTHONIOENCODING=utf-8 python rejim.py --test kontrol
    PYTHONIOENCODING=utf-8 python rejim.py --test sikisma
    PYTHONIOENCODING=utf-8 python rejim.py --test plasebo     (kontrol grubu: rastgele giris)
    PYTHONIOENCODING=utf-8 python rejim.py --test volrejim    (islem bazli + portfoy bazli)
    PYTHONIOENCODING=utf-8 python rejim.py --test volportfoy
    PYTHONIOENCODING=utf-8 python rejim.py --test saat
    PYTHONIOENCODING=utf-8 python rejim.py --test valid       (secilen 3 aday, TEK sefer)
    PYTHONIOENCODING=utf-8 python rejim.py --test hepsi --n 10

ONEMLI: saglam(T) ve sig_plasebo() baska ailelerden import edilebilir. Gunluk altcoin evreninde
ham ort_net (ortalama R) TEK BASINA anlamsizdir - rastgele giris de pozitif verir. Her aday icin
eslesmis plasebo + 3R-winsorize ortalama sart.

Kurallar: veri/sim SADECE harness.py; sinyal t kapanisinda, giris t+1 acilisinda (piyasa);
tum gostergeler geriye bakar (rolling, gelecek bar yok). Maliyet harness varsayilani.
"""
import argparse, json, os, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

AILE = "rejim"
LAB = os.path.dirname(os.path.abspath(__file__))
SONUC = os.path.join(LAB, "sonuc_rejim.json")


# ------------------------------------------------------------------ gostergeler (hepsi gecmise bakar)
def _roll_mean(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.concatenate([[0.0], np.cumsum(x)])
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _roll_std(x, n):
    m = _roll_mean(x, n); m2 = _roll_mean(np.asarray(x, float) ** 2, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _roll_max(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(np.asarray(x, float), n).max(axis=1)
    return out


def _roll_min(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(np.asarray(x, float), n).min(axis=1)
    return out


def _roll_q(x, n, q):
    """son n barin (i dahil) q-kantili"""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(~np.isnan(x))
    if len(ok) == 0:
        return out
    f = ok[0]; y = x[f:]
    if len(y) >= n:
        out[f + n - 1:] = np.quantile(sliding_window_view(y, n), q, axis=1)
    return out


def _shift1(x):
    o = np.full(len(x), np.nan); o[1:] = x[:-1]; return o


def atr(bars, n=14):
    h, l, c = bars[:, 2], bars[:, 3], bars[:, 4]
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return _roll_mean(tr, n)


def rsi(c, n=2):
    d = np.diff(c, prepend=c[0])
    au = _roll_mean(np.where(d > 0, d, 0.0), n)
    ad = _roll_mean(np.where(d < 0, -d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = au / np.where(ad > 0, ad, np.nan)
    out = np.where(np.isnan(rs), np.where(au > 0, 100.0, 50.0), 100 - 100 / (1 + rs))
    out[np.isnan(au)] = np.nan
    return out


def bollinger(c, n=20, k=2.0):
    m = _roll_mean(c, n); s = _roll_std(c, n)
    return m, m + k * s, m - k * s


def _ilk_sonra(idx_true, i, cap):
    """idx_true: kosulun saglandigi bar indeksleri (sirali). i'den SONRAKI ilk indeks, cap ile sinirli."""
    p = np.searchsorted(idx_true, i, side="right")
    if p >= len(idx_true):
        return None
    j = int(idx_true[p])
    return j if j <= i + cap else None


def chandelier(bars, i, side, k, a, cap):
    """girisden (i+1) itibaren ATR iz suren cikis bari; sadece gecmis barlari kullanir"""
    h, l, c = bars[:, 2], bars[:, 3], bars[:, 4]
    n = len(c); ext = None
    for j in range(i + 1, min(n, i + 1 + cap)):
        ext = h[j] if ext is None else (max(ext, h[j]) if side == 1 else min(ext, l[j]))
        if ext is None:
            continue
        if side == -1 and j == i + 1:
            ext = l[j]
        av = a[j]
        if np.isnan(av):
            continue
        if side == 1 and c[j] < ext - k * av:
            return j
        if side == -1 and c[j] > ext + k * av:
            return j
    return None


# ------------------------------------------------------------------ veri
def bar_yukle(sym, tf):
    if tf == "1d":
        return H.veri_1d(sym)
    a = H.veri_1h(sym)
    return a if tf == "1h" else H.resample(a, tf)


def _kaynak(tf):
    return "1d" if tf == "1d" else "1h"


def raporla(ad, params, islemler, tf, yaz=True, not_=""):
    kay = _kaynak(tf)
    tr = H.rapor(islemler, "train", kay); va = None
    out = dict(ad=ad, params=params, train=tr, valid=None)
    if yaz:
        H.kaydet(AILE, ad, params, "train", tr, not_)
    return out


# ================================================================== 1) KONTROL: mean-reversion
def sig_kontrol(bars, mode="rsi2", tf="1h", atr_n=14, atr_k=2.0, max_days=5, allow_short=True,
                rsi_al=10, rsi_sat=90):
    c = bars[:, 4]; n = len(c)
    if n < 120:
        return []
    mid, up, lo = bollinger(c, 20, 2.0)
    a = atr(bars, atr_n); r = rsi(c, 2)
    cap = max(2, int(max_days * 24 / H.TF_H[tf]))
    iu = np.flatnonzero(c >= mid)   # long cikis (orta bant)
    idn = np.flatnonzero(c <= mid)  # short cikis
    if mode == "rsi2":
        L = (r < rsi_al); S = (r > rsi_sat)
    else:
        L = (c < lo); S = (c > up)
    if not allow_short:
        S = np.zeros(n, bool)
    ok = ~np.isnan(a) & ~np.isnan(mid) & (a > 0)
    sig = []
    for i in np.flatnonzero((L | S) & ok):
        i = int(i)
        if i < 60 or i >= n - 1:
            continue
        side = 1 if L[i] else -1
        stop = float(c[i] - side * atr_k * a[i])
        ci = _ilk_sonra(iu if side == 1 else idn, i, cap)
        sig.append(dict(i=i, side=side, entry=None, stop=stop, tp=None, cikis_i=ci, max_days=max_days))
    return sig


# ================================================================== 2) Sikisma kirilimi
def sig_sikisma(bars, tf="4h", bw_pct=0.20, bw_look=100, brk_n=20, trail_k=3.0, atr_n=14,
                max_days=20, allow_short=True, min_sp=0.0):
    h, l, c = bars[:, 2], bars[:, 3], bars[:, 4]; n = len(c)
    if n < bw_look + 60:
        return []
    mid, up, lo = bollinger(c, 20, 2.0)
    bw = (up - lo) / mid
    thr = _roll_q(bw, bw_look, min(bw_pct, 1.0))
    a = atr(bars, atr_n)
    pmax = _shift1(_roll_max(h, brk_n)); pmin = _shift1(_roll_min(l, brk_n))
    cap = max(2, int(max_days * 24 / H.TF_H[tf]))
    sqz = (~np.isnan(thr)) & (bw <= thr)
    L = sqz & (~np.isnan(pmax)) & (c > pmax)
    S = sqz & (~np.isnan(pmin)) & (c < pmin) if allow_short else np.zeros(n, bool)
    ok = (~np.isnan(a)) & (a > 0) & (~np.isnan(mid))
    sig = []
    for i in np.flatnonzero((L | S) & ok):
        i = int(i)
        if i >= n - 1:
            continue
        side = 1 if L[i] else -1
        stop = float(lo[i]) if side == 1 else float(up[i])
        sp = abs(c[i] - stop) / c[i]
        if min_sp > 0 and sp < min_sp:                       # cok dar stop -> ATR tabani
            stop = float(c[i] - side * min_sp * c[i])
        if (side == 1 and stop >= c[i]) or (side == -1 and stop <= c[i]):
            continue
        ci = chandelier(bars, i, side, trail_k, a, cap)
        sig.append(dict(i=i, side=side, entry=None, stop=stop, tp=None, cikis_i=ci, max_days=max_days))
    return sig


def sig_plasebo(bars, tf="1d", oran=0.02, trail_k=3.0, atr_n=14, max_days=120, tohum=0,
                allow_short=True, stop_atr=None):
    """PLASEBO: sinyal filtresi YOK, rastgele barlarda giris. Ayni stop (karsi bant) + ayni
    ATR iz suren cikis. Kirilim+sikisma filtresinin gercekten bir sey ekleyip eklemedigini olcer."""
    c = bars[:, 4]; n = len(c)
    if n < 160:
        return []
    mid, up, lo = bollinger(c, 20, 2.0)
    a = atr(bars, atr_n)
    cap = max(2, int(max_days * 24 / H.TF_H[tf]))
    rng = np.random.default_rng(tohum + (int(bars[0, 0]) % 100000))
    ok = np.flatnonzero((~np.isnan(a)) & (a > 0) & (~np.isnan(mid)) & (np.arange(n) < n - 1) & (np.arange(n) > 120))
    if len(ok) == 0:
        return []
    sec = np.sort(rng.choice(ok, size=max(1, int(len(ok) * oran)), replace=False))
    sides = rng.choice([1, -1], size=len(sec)) if allow_short else np.ones(len(sec), int)
    sig = []
    for i, side in zip(sec, sides):
        i = int(i); side = int(side)
        stop = float(lo[i]) if side == 1 else float(up[i])
        if stop_atr:
            stop = float(c[i] - side * stop_atr * a[i])
        if (side == 1 and stop >= c[i]) or (side == -1 and stop <= c[i]):
            continue
        ci = chandelier(bars, i, side, trail_k, a, cap)
        sig.append(dict(i=i, side=side, entry=None, stop=stop, tp=None, cikis_i=ci, max_days=max_days))
    return sig


def sig_kirilim_saf(bars, tf="1d", brk_n=20, trail_k=3.0, atr_n=14, max_days=120, allow_short=True):
    """Sikisma filtresi OLMADAN sadece N-bar kirilim (filtrenin katkisini olcmek icin)."""
    return sig_sikisma(bars, tf=tf, bw_pct=1.01, bw_look=100, brk_n=brk_n, trail_k=trail_k,
                       atr_n=atr_n, max_days=max_days, allow_short=allow_short)


# ================================================================== 3) Vol rejimi overlay
def realized_vol(c, n=30):
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=0.0)
    lr[0] = 0.0
    return _roll_std(lr, n) * np.sqrt(365)


def genisleyen_tercil(v, min_gecmis=365):
    """v[i] icin, SADECE i'ye kadarki gecmisin 33/67 kantilleri -> rejim 0/1/2 (dusuk/orta/yuksek)"""
    n = len(v); reg = np.full(n, -1, int)
    ok = np.flatnonzero(~np.isnan(v))
    if len(ok) == 0:
        return reg
    buf = []; q1 = q2 = None; son = -10 ** 9
    for i in range(n):
        if not np.isnan(v[i]):
            if len(buf) >= min_gecmis:
                if i - son >= 21 or q1 is None:      # esikler 21 barda bir yenilenir (hep GECMISTEN)
                    q1, q2 = np.quantile(buf, [1 / 3, 2 / 3]); son = i
                reg[i] = 0 if v[i] <= q1 else (2 if v[i] >= q2 else 1)
            buf.append(v[i])
    return reg


def btc_vol_rejim(n=30, min_gecmis=365):
    a = H.veri_1d("BTCUSDT")
    v = realized_vol(a[:, 4], n)
    return a[:, 0], genisleyen_tercil(v, min_gecmis), v


def sig_trend_1d(bars, ma=50, atr_n=14, atr_k=2.0, max_days=200):
    c = bars[:, 4]; n = len(c)
    if n < ma + 60:
        return []
    m = _roll_mean(c, ma); a = atr(bars, atr_n)
    L = (c > m) & (_shift1(c) <= _shift1(m))
    idn = np.flatnonzero(c < m)
    ok = (~np.isnan(a)) & (a > 0) & (~np.isnan(m))
    sig = []
    for i in np.flatnonzero(L & ok):
        i = int(i)
        if i < ma + 5 or i >= n - 1:
            continue
        ci = _ilk_sonra(idn, i, max_days)
        sig.append(dict(i=i, side=1, entry=None, stop=float(c[i] - atr_k * a[i]), tp=None,
                        cikis_i=ci, max_days=max_days))
    return sig


# ================================================================== yardimcilar
def kosu(syms, tf, sig_fn, **kw):
    T = []
    for s in syms:
        try:
            b = bar_yukle(s, tf)
        except Exception:
            continue
        if len(b) < 150:
            continue
        sg = sig_fn(b, tf=tf, **kw) if "tf" in sig_fn.__code__.co_varnames else sig_fn(b, **kw)
        if not sg:
            continue
        for t in H.sim_islem(b, sg, tf=tf):
            t["sym"] = s
            T.append(t)
    return T


def saglam(T):
    """Agir kuyruk teshisi: ort_net cok az sayida dev kazanca bagli mi?"""
    F = [t for t in T if t["res"] != "cancelled"]
    if len(F) < 20:
        return dict(n=len(F))
    x = np.array([t["net"] for t in F])
    xs = np.sort(x)[::-1]
    k = max(1, int(len(x) * 0.01))
    return dict(n=len(x), ort=round(float(x.mean()), 4), medyan=round(float(np.median(x)), 4),
                wins10=round(float(np.clip(x, -1.5, 10).mean()), 4),
                wins3=round(float(np.clip(x, -1.5, 3).mean()), 4),
                top1_pay=round(float(xs[:k].sum() / max(1e-9, abs(x.sum()))), 3) if x.sum() != 0 else None,
                maxR=round(float(x.max()), 1))


def ozet(r):
    if not r or r.get("ort_net") is None:
        return f"n={r.get('n', 0)} (yetersiz)"
    return (f"n={r['n']} ort_net={r['ort_net']:+.4f}R ga99=[{r['ga99'][0]:+.3f},{r['ga99'][1]:+.3f}] "
            f"win={r['win']:.2f} y1={r['yari1']:+.3f} y2={r['yari2']:+.3f}")


# ================================================================== testler
def test_kontrol(syms1h, N=None):
    print("\n=== 1) KONTROL: mean-reversion (NEGATIF bekleniyor) ===")
    res = []
    for tf in ("1h", "4h"):
        for mode in ("rsi2", "bb"):
            for short in (True, False):
                ad = f"kontrol_{mode}_{tf}_{'ls' if short else 'l'}"
                p = dict(mode=mode, tf=tf, atr_k=2.0, max_days=5 if tf == "1h" else 10, allow_short=short)
                t0 = time.time()
                T = kosu(syms1h, tf, sig_kontrol, mode=mode, atr_k=2.0,
                         max_days=p["max_days"], allow_short=short)
                tr = H.rapor(T, "train", "1h")
                H.kaydet(AILE, ad, p, "train", tr, "kontrol ailesi - harness yanlilik testi")
                print(f"  {ad:28s} {ozet(tr)}  [{time.time()-t0:.0f}s]")
                res.append((ad, p, tr, T))
    return res


def test_sikisma(syms1h, syms1d):
    print("\n=== 2) Sikisma kirilimi (BB genislik alt %20 + N-bar kirilim) ===")
    res = []
    grid4 = [dict(bw_pct=0.20, brk_n=20, trail_k=3.0), dict(bw_pct=0.20, brk_n=20, trail_k=2.0),
             dict(bw_pct=0.20, brk_n=55, trail_k=3.0), dict(bw_pct=0.10, brk_n=20, trail_k=3.0),
             dict(bw_pct=0.30, brk_n=20, trail_k=3.0), dict(bw_pct=0.20, brk_n=10, trail_k=3.0),
             dict(bw_pct=0.20, brk_n=20, trail_k=3.0, min_sp=0.015),
             dict(bw_pct=0.20, brk_n=20, trail_k=3.0, allow_short=False)]
    for g in grid4:
        ad = "sqz_4h_" + "_".join(f"{k}{v}" for k, v in g.items())
        p = dict(tf="4h", max_days=20, **g)
        t0 = time.time()
        T = kosu(syms1h, "4h", sig_sikisma, max_days=20, **g)
        tr = H.rapor(T, "train", "1h")
        H.kaydet(AILE, ad, p, "train", tr)
        print(f"  {ad:52s} {ozet(tr)}  [{time.time()-t0:.0f}s]")
        res.append((ad, p, tr, T))
    grid1 = [dict(bw_pct=0.20, brk_n=20, trail_k=3.0), dict(bw_pct=0.20, brk_n=55, trail_k=3.0),
             dict(bw_pct=0.10, brk_n=20, trail_k=3.0), dict(bw_pct=0.30, brk_n=20, trail_k=3.0),
             dict(bw_pct=0.20, brk_n=20, trail_k=2.0), dict(bw_pct=0.20, brk_n=20, trail_k=4.0),
             dict(bw_pct=0.20, brk_n=20, trail_k=3.0, allow_short=False)]
    for g in grid1:
        ad = "sqz_1d_" + "_".join(f"{k}{v}" for k, v in g.items())
        p = dict(tf="1d", max_days=120, **g)
        t0 = time.time()
        T = kosu(syms1d, "1d", sig_sikisma, max_days=120, **g)
        tr = H.rapor(T, "train", "1d")
        H.kaydet(AILE, ad, p, "train", tr)
        print(f"  {ad:52s} {ozet(tr)}  [{time.time()-t0:.0f}s]")
        res.append((ad, p, tr, T))
    return res


def test_plasebo(syms1h, syms1d):
    """Sikisma sonuclari icin kontrol grubu: (a) rastgele giris, (b) filtresiz kirilim."""
    print("\n=== 2b) PLASEBO / baseline (sikisma edge'i gercek mi?) ===")
    res = []
    isler = [
        ("plasebo_1d_rastgele_ls", syms1d, "1d", sig_plasebo, dict(oran=0.02, trail_k=3.0, max_days=120)),
        ("plasebo_1d_rastgele_l", syms1d, "1d", sig_plasebo, dict(oran=0.02, trail_k=3.0, max_days=120, allow_short=False)),
        ("kirilim_saf_1d_n20_k3", syms1d, "1d", sig_kirilim_saf, dict(brk_n=20, trail_k=3.0, max_days=120)),
        ("kirilim_saf_1d_n20_k3_l", syms1d, "1d", sig_kirilim_saf, dict(brk_n=20, trail_k=3.0, max_days=120, allow_short=False)),
        ("plasebo_4h_rastgele_ls", syms1h, "4h", sig_plasebo, dict(oran=0.02, trail_k=2.0, max_days=20)),
        ("kirilim_saf_4h_n20_k2", syms1h, "4h", sig_kirilim_saf, dict(brk_n=20, trail_k=2.0, max_days=20)),
    ]
    for ad, syms, tf, fn, kw in isler:
        t0 = time.time()
        T = kosu(syms, tf, fn, **kw)
        tr = H.rapor(T, "train", _kaynak(tf))
        H.kaydet(AILE, ad, dict(tf=tf, **kw), "train", tr, "plasebo/baseline")
        print(f"  {ad:28s} {ozet(tr)}  [{time.time()-t0:.0f}s]")
        res.append((ad, dict(tf=tf, **kw), tr, T))
    return res


def test_komsu4h(syms1h):
    print("\n=== 2c) 4h trail_k komsulari ===")
    res = []
    for k in (1.5, 2.0, 2.5):
        ad = f"sqz_4h_bw0.2_n20_k{k}"
        t0 = time.time()
        T = kosu(syms1h, "4h", sig_sikisma, bw_pct=0.20, brk_n=20, trail_k=k, max_days=20)
        tr = H.rapor(T, "train", "1h")
        H.kaydet(AILE, ad, dict(tf="4h", bw_pct=0.2, brk_n=20, trail_k=k), "train", tr, "komsu taramasi")
        print(f"  {ad:28s} {ozet(tr)}  [{time.time()-t0:.0f}s]")
        res.append((ad, dict(tf="4h", bw_pct=0.2, brk_n=20, trail_k=k), tr, T))
    return res


def test_volrejim(syms1d):
    print("\n=== 3) Vol rejimi overlay (gunluk MA50 trend, 30g gerceklesen vol tercili) ===")
    res = []
    ots, breg, bvol = btc_vol_rejim(30, 365)
    ridx = {int(t): int(r) for t, r in zip(ots, breg)}
    okey = np.array(sorted(ridx))

    def btc_reg_at(t_ms):
        k = np.searchsorted(okey, t_ms - 86400_000, side="right") - 1
        return ridx[int(okey[k])] if k >= 0 else -1

    for atr_k in (2.0, 3.0):
        for ma in (50, 100):
            base = f"trend_ma{ma}_atr{atr_k}"
            t0 = time.time()
            T = []
            per = {}
            for s in syms1d:
                try:
                    b = H.veri_1d(s)
                except Exception:
                    continue
                if len(b) < 400:
                    continue
                sg = sig_trend_1d(b, ma=ma, atr_k=atr_k)
                if not sg:
                    continue
                cv = realized_vol(b[:, 4], 30)
                creg = genisleyen_tercil(cv, 365)
                mp = {int(b[i, 0]) + 86400_000: int(creg[i]) for i in range(len(b))}
                for t in H.sim_islem(b, sg, tf="1d"):
                    t["sym"] = s
                    t["breg"] = btc_reg_at(t["t"])
                    t["creg"] = mp.get(int(t["t"]), -1)
                    T.append(t)
            tr = H.rapor(T, "train", "1d")
            H.kaydet(AILE, base + "_hepsi", dict(ma=ma, atr_k=atr_k, rejim="hepsi"), "train", tr)
            print(f"  {base+'_hepsi':34s} {ozet(tr)}  [{time.time()-t0:.0f}s]")
            res.append((base + "_hepsi", dict(ma=ma, atr_k=atr_k, rejim="hepsi"), tr, T))
            for key, adk in (("breg", "btcvol"), ("creg", "coinvol")):
                for rg, nm in ((0, "dusuk"), (1, "orta"), (2, "yuksek")):
                    sub = [t for t in T if t.get(key) == rg]
                    r = H.rapor(sub, "train", "1d")
                    ad = f"{base}_{adk}_{nm}"
                    H.kaydet(AILE, ad, dict(ma=ma, atr_k=atr_k, rejim=f"{adk}:{nm}"), "train", r)
                    print(f"    {ad:36s} {ozet(r)}")
                    res.append((ad, dict(ma=ma, atr_k=atr_k, rejim=f"{adk}:{nm}"), r, sub))
    return res


def test_volportfoy(min_gun=400):
    """Q3 PORTFOY versiyonu: R-carpani kuyruk artefaktindan bagimsiz.
    Esit agirlikli long: kapanis > MA50. Vol rejimi overlay ile filtrelenir."""
    print("\n=== 3b) Vol rejimi overlay - PORTFOY (esit agirlikli MA50 long) ===")
    syms, gun, O, C, V = H.gunluk_panel(min_gun=min_gun)
    n, m = C.shape
    print(f"  panel: {n} gun x {m} coin")
    ots, breg, bvol = btc_vol_rejim(30, 365)
    rmap = {int(t): int(r) for t, r in zip(ots, breg)}
    reg_g = np.array([rmap.get(int(g), -1) for g in gun])
    MA = np.full((n, m), np.nan)
    for j in range(m):
        c = C[:, j]
        ok = ~np.isnan(c)
        if ok.sum() < 60:
            continue
        f = int(np.flatnonzero(ok)[0])
        MA[f:, j] = _roll_mean(np.nan_to_num(c[f:], nan=0.0), 50)
    aktif = (C > MA) & ~np.isnan(MA) & ~np.isnan(C)
    say = aktif.sum(axis=1)
    baz = np.zeros((n, m))
    for t in range(n):
        if say[t] > 0:
            baz[t, aktif[t]] = 1.0 / max(say[t], 20)
    res = []
    # kiyaslar
    Wb = np.zeros((n, m))
    if "BTCUSDT" in syms:
        Wb[:, syms.index("BTCUSDT")] = 1.0
    Wall = np.where(~np.isnan(C), 1.0, 0.0); Wall = Wall / np.maximum(Wall.sum(axis=1, keepdims=True), 20)
    for ad, W in (("kiyas_btc_altut", Wb), ("kiyas_esit_altut", Wall), ("ma50_long_hepsi", baz)):
        eq, g = H.sim_portfoy(O, C, W)
        for bl in ("train",):
            r = H.portfoy_rapor(eq, g, gun[:len(g)], bl, "1d")
            H.kaydet(AILE, ad, dict(tip="portfoy", min_gun=min_gun), bl, r)
            print(f"  {ad:26s} {bl:6s} {r}")
        res.append((ad, eq, g))
    for rg, nm in ((0, "dusuk"), (1, "orta"), (2, "yuksek")):
        W = baz * (reg_g[:, None] == rg)
        eq, g = H.sim_portfoy(O, C, W)
        ad = f"ma50_long_btcvol_{nm}"
        for bl in ("train",):
            r = H.portfoy_rapor(eq, g, gun[:len(g)], bl, "1d")
            H.kaydet(AILE, ad, dict(tip="portfoy", rejim=nm, min_gun=min_gun), bl, r)
            print(f"  {ad:26s} {bl:6s} {r}")
        res.append((ad, eq, g))
    # dusuk+orta (yuksek vol disla)
    W = baz * (reg_g[:, None] != 2)
    eq, g = H.sim_portfoy(O, C, W)
    for bl in ("train",):
        r = H.portfoy_rapor(eq, g, gun[:len(g)], bl, "1d")
        H.kaydet(AILE, "ma50_long_btcvol_yuksek_haric", dict(tip="portfoy", rejim="0+1", min_gun=min_gun), bl, r)
        print(f"  {'ma50_long_yuksek_haric':26s} {bl:6s} {r}")
    return res


def test_saat(syms1h):
    print("\n=== 4) Gun-ici saat / hafta gunu etkisi (1h, esit agirlikli piyasa serisi) ===")
    ser = {}
    for s in syms1h:
        try:
            a = H.veri_1h(s)
        except Exception:
            continue
        if len(a) < 500:
            continue
        c = a[:, 4]
        r = np.full(len(c), np.nan); r[1:] = c[1:] / c[:-1] - 1
        for t, x in zip(a[:, 0], r):
            if not np.isnan(x) and abs(x) < 0.5:
                ser.setdefault(int(t), []).append(x)
    ots = np.array(sorted(ser))
    mkt = np.array([np.mean(ser[int(t)]) for t in ots])
    say = np.array([len(ser[int(t)]) for t in ots])
    ots = ots[say >= 20]; mkt = mkt[say >= 20]
    tr_m = ots < H.T_VALID_1H
    hh = ((ots // 3600_000) % 24).astype(int)
    dd = (((ots // 86400_000) + 3) % 7).astype(int)   # 0=Pzt
    print(f"  piyasa serisi: {len(ots)} saat, train {tr_m.sum()}")
    sonuc = []
    for etiket, grp, K in (("saat", hh, 24), ("gun", dd, 7)):
        satir = []
        for k in range(K):
            m = tr_m & (grp == k)
            x = mkt[m]
            if len(x) < 30:
                continue
            t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            satir.append((k, len(x), x.mean(), t))
        satir.sort(key=lambda z: -abs(z[3]))
        # Bonferroni: |t| > z_{1-0.025/K}
        from math import erf, sqrt
        kritik = {24: 3.21, 7: 2.69}[K]   # iki yonlu 5%, Bonferroni K test
        print(f"  --- {etiket} (Bonferroni kritik |t|~{kritik}) ---")
        for k, n, m_, t in satir[:5]:
            print(f"    {etiket}={k:2d} n={n:5d} ort={m_*1e4:+7.2f}bp t={t:+6.2f} {'GECTI' if abs(t)>kritik else ''}")
        gecen = [z for z in satir if abs(z[3]) > kritik]
        rap = dict(n=len(satir), en_iyi=[dict(k=int(k), n=int(n), ort_bp=round(m_ * 1e4, 2), t=round(t, 2))
                                         for k, n, m_, t in satir[:5]], bonferroni_gecen=len(gecen))
        H.kaydet(AILE, f"mevsimsellik_{etiket}", dict(grup=etiket, K=K, kritik=kritik,
                 test_sayisi=K), "train", rap, "istatistik taramasi (islem simulasyonu degil)")
        sonuc.append((etiket, satir, gecen))
    # saat x gun (168 test)
    kritik = 3.83
    satir = []
    for k in range(24):
        for d in range(7):
            m = tr_m & (hh == k) & (dd == d)
            x = mkt[m]
            if len(x) < 30:
                continue
            t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            satir.append(((d, k), len(x), x.mean(), t))
    satir.sort(key=lambda z: -abs(z[3]))
    gecen = [z for z in satir if abs(z[3]) > kritik]
    print(f"  --- saat x gun (168 test, Bonferroni kritik |t|~{kritik}) ---")
    for kd, n, m_, t in satir[:5]:
        print(f"    gun={kd[0]} saat={kd[1]:2d} n={n:4d} ort={m_*1e4:+7.2f}bp t={t:+6.2f} {'GECTI' if abs(t)>kritik else ''}")
    H.kaydet(AILE, "mevsimsellik_saatxgun", dict(grup="saatxgun", K=168, kritik=kritik), "train",
             dict(n=len(satir), en_iyi=[dict(gun=int(z[0][0]), saat=int(z[0][1]), n=int(z[1]),
                  ort_bp=round(z[2] * 1e4, 2), t=round(z[3], 2)) for z in satir[:5]],
                  bonferroni_gecen=len(gecen)),
             "istatistik taramasi")
    sonuc.append(("saatxgun", satir, gecen))
    return sonuc



# ================================================================== VALID (TEK SEFER)
ADAYLAR = [
    ("A_sqz4h_bw0.2_n20_k2.0_ls", "4h", "sikisma",
     dict(bw_pct=0.20, brk_n=20, trail_k=2.0, max_days=20, allow_short=True)),
    ("B_sqz4h_bw0.2_n20_k2.0_shortonly", "4h", "sikisma_short",
     dict(bw_pct=0.20, brk_n=20, trail_k=2.0, max_days=20)),
    ("C_brk1d_n20_k3.0_longonly", "1d", "kirilim_saf",
     dict(brk_n=20, trail_k=3.0, max_days=120, allow_short=False)),
]
KONTROLLER = [
    ("K_plasebo4h_k2.0_ls", "4h", "plasebo", dict(oran=0.02, trail_k=2.0, max_days=20)),
    ("K_plasebo1d_k3.0_l", "1d", "plasebo", dict(oran=0.02, trail_k=3.0, max_days=120, allow_short=False)),
]


def sig_sikisma_short(bars, **kw):
    sg = sig_sikisma(bars, **kw)
    return [s for s in sg if s["side"] == -1]


FN = dict(sikisma=sig_sikisma, sikisma_short=sig_sikisma_short, kirilim_saf=sig_kirilim_saf, plasebo=sig_plasebo)


def test_valid(s1h, s1d):
    print("\n=== VALID (TEK SEFER, 3 aday + 2 kontrol) ===")
    out = []
    for ad, tf, fn, kw in ADAYLAR + KONTROLLER:
        syms = s1d if tf == "1d" else s1h
        T = kosu(syms, tf, FN[fn], **kw)
        kay = _kaynak(tf)
        tr = H.rapor(T, "train", kay); va = H.rapor(T, "valid", kay)
        dtr = saglam(H.bolum_filtre(T, "train", kay)); dva = saglam(H.bolum_filtre(T, "valid", kay))
        aday = ad.startswith(("A_", "B_", "C_"))
        H.kaydet(AILE, ad, dict(tf=tf, fn=fn, **kw), "valid", dict(va, saglam=dva),
                 "ADAY - valid tek sefer" if aday else "KONTROL grubu (aday degil)")
        ok, sebep = H.kabul(tr, va)
        print(f"  {ad}")
        print(f"    train {ozet(tr)} wins3R={dtr.get('wins3')}")
        print(f"    valid {ozet(va)} wins3R={dva.get('wins3')}")
        print(f"    kabul={ok} ({sebep})")
        out.append(dict(ad=ad, params=dict(tf=tf, fn=fn, **kw), train=tr, valid=va,
                        train_saglam=dtr, valid_saglam=dva, kabul=bool(ok), kabul_sebep=sebep, aday=aday))
    with open(os.path.join(LAB, "valid_rejim_ham.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return out


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="hepsi",
                    choices=["kontrol", "sikisma", "plasebo", "volrejim", "volportfoy", "saat", "hepsi", "valid"])
    ap.add_argument("--n", type=int, default=0, help="coin sayisi limiti (0=hepsi)")
    a = ap.parse_args()
    s1h = H.semboller_1h(); s1d = H.semboller_1d()
    if a.n:
        s1h = s1h[:a.n]; s1d = s1d[:a.n]
    print(f"evren: 1h={len(s1h)} coin, 1d={len(s1d)} coin")
    if a.test in ("kontrol", "hepsi"):
        test_kontrol(s1h)
    if a.test in ("sikisma", "hepsi"):
        test_sikisma(s1h, s1d)
    if a.test in ("plasebo", "sikisma", "hepsi"):
        test_plasebo(s1h, s1d)
        test_komsu4h(s1h)
    if a.test in ("volrejim", "hepsi"):
        test_volrejim(s1d)
    if a.test in ("volportfoy", "volrejim", "hepsi"):
        test_volportfoy()
    if a.test in ("saat", "hepsi"):
        test_saat(s1h)
    if a.test == "valid":
        test_valid(s1h, s1d)


if __name__ == "__main__":
    main()
