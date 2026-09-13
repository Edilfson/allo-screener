"""ict_ozellik - "ICT'de kazananlara bakip eksikleri duzeltebilir miyiz?" sorusunun dogru yontemle testi.

Soru: GIRISTEN ONCE bilinen ozellikler kazanani (TP) kaybedenden ayiriyor mu, ve bu ayrim RASTGELE
girislerde (plasebo) de ayni kolaylikla bulunuyor mu?

ASAMALAR (argparse --asama):
  uret        : 1h/4h/6h/12h/1d x 98 coin x long+short. ICT (ict2 C0_base esdegeri = bt.detect) + plasebo
                P1 (ict2 C1c: ICT'nin ayni barlari, rastgele yon) + P2 (ict2 C1: rastgele bar + rastgele yon),
                N_SEED seed. Her dolan islem icin ~30 ozellik (sinyal bari kapanisinda bilinen).
                Dilim basina ara sonuc: <SCRATCH>/ict_ozellik/<onek>_veri_<tf>.npz
  esdeger     : kendi ICT sinyal listemizin bt.detect PASS kumesi ve ict2._sinyaller C0_base ile birebir ayni oldugunu dogrular
  nedensellik : >=50 rastgele islemde veriyi sinyal barinda kesip ozellikleri yeniden hesaplar (ayni cikmali)
  madencilik  : SADECE TRAIN (kaynak=1h, < 2025-07-01). 4 zaman-sirali kat, 7 gun bosluk. Tek degiskenli dilim,
                L2 lojistik, derinlik 2/3 agac (min yaprak 100), model-skoru filtreleri (%30/%50) ve basit kurallar
                (<=3 esik). Ayni pipeline P1/P2 her seed + 100 etiket(sonuc) permutasyonu. Aday kurali.
  valid       : adaylar (en fazla 3) VALID'de TEK SEFER (ayni plasebo karsilastirmasi).
  hepsi       : sirayla hepsi.

Kullanim: PYTHONIOENCODING=utf-8 python ict_ozellik.py --asama hepsi --coin 98
          (duman testi: --coin 10 --onek duman)
"""
import os, sys, json, argparse, hashlib, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
ARACLAR = os.path.dirname(LAB)
sys.path.insert(0, LAB); sys.path.insert(0, ARACLAR)
import harness as H
import bt
import ict2

AILE = "ict_ozellik"
P0 = dict(bt.P0)                      # DEGISTIRILMEZ (canli ICT kurallari)
TFS = ["1h", "4h", "6h", "12h", "1d"]
N_SEED = 20
N_PERM = 100
DAY = 86400_000
HOUR = 3600_000
GAP = 7 * DAY
ARA = os.path.join(H.SCRATCH, "ict_ozellik")
WINS = (-1.5, 3.0)                    # 3R-winsorize (rejim.saglam wins3 ile ayni)

OZ = ["stop_pct", "atr_pct", "giris_mes_r", "bos_yas", "disp_atr", "ob_atr", "fvg_atr",
      "hacim_sinyal", "hacim_bos", "rsi14_hiz", "ema20_egim_hiz", "ema50_egim_hiz",
      "ema50_mes_hiz", "ema200_mes_hiz", "coin_ma50_hiz", "coin_ma50_mes_hiz",
      "btc_ma50_hiz", "btc_ma50_mes_hiz", "btc_7g_hiz", "coin_7g_hiz", "coin_30g_hiz",
      "vol_30g", "fitil_lehine", "fitil_aleyhine", "swing_aralik_atr", "saat_utc",
      "hafta_gunu", "dilim_saat", "yon", "likidite_sira"]
OZ_ACIKLAMA = ("'_hiz' = yon-hizali (deger * yon): pozitif = islem yonune uyumlu (long icin yukari trend/ustte, "
               "short icin asagi trend/altta). rsi14_hiz=(RSI-50)*yon. fitil_lehine: long icin alt fitil, short icin ust. "
               "bos_yas/disp_atr/ob_atr/fvg_atr/hacim_bos: sinyal barindan geriye bakan GENEL yapi taramasi "
               "(ICT islemlerinde ICT'nin kendi BOS/FVG/OB degerleri; plasebo islemlerinde ayni tarama, yoksa NaN).")
# meta kolonlari
M = dict(uni=0, seed=1, sym=2, tfh=3, t=4, side=5, res=6, r=7, net=8, sp=9, i=10, entry=11, stop=12)
RES = {"stopped": 0, "target_done": 1, "timeout": 2, "cikis": 3}


# =============================================================== ozellik cikarimi
def _rsi(c, n=14):
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c); up = np.maximum(d, 0); dn = np.maximum(-d, 0)
    au = up[:n].mean(); ad = dn[:n].mean()
    out[n] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    for k in range(n + 1, len(c)):
        au = (au * (n - 1) + up[k - 1]) / n; ad = (ad * (n - 1) + dn[k - 1]) / n
        out[k] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    return out


def _gunluk_ma(a, N=50):
    """gunluk dizi -> (ot, c, ma50)"""
    c = a[:, 4]
    m = np.convolve(c, np.ones(N) / N, "full")[:len(c)]
    m[:N - 1] = np.nan
    return a[:, 0], c, m


def coin_gunluk(sym, a1h):
    """ict2._coin_rejim ile ayni kaynak secimi (dosya varsa veri_1d, yoksa 1h resample)"""
    try:
        return H.veri_1d(sym)
    except Exception:
        return H.resample(a1h, "1d")


def gunluk_qv(a1h):
    """1h -> (gun_ot, gunluk quote hacim) sadece tam gunler"""
    g = (a1h[:, 0] // DAY).astype(np.int64)
    qv = a1h[:, 5] * a1h[:, 4]
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    s = np.bincount(inv, weights=qv)
    ok = cnt == 24
    return u[ok] * DAY, s[ok]


def likidite_panel(qv_dict):
    """sym -> (gun_ot, qv) -> sym -> (gun_ot, kesitsel sira [0..1]) ; 30g ort quote hacim (sadece gecmis)"""
    roll = {}
    for s, (ot, q) in qv_dict.items():
        if len(q) < 30:
            continue
        cs = np.cumsum(np.concatenate([[0.0], q]))
        m = (cs[30:] - cs[:-30]) / 30
        roll[s] = (ot[29:], m)
    if not roll:
        return {}
    gun = np.unique(np.concatenate([v[0] for v in roll.values()]))
    gi = {int(g): k for k, g in enumerate(gun)}
    syms = sorted(roll)
    X = np.full((len(gun), len(syms)), np.nan)
    for j, s in enumerate(syms):
        ot, m = roll[s]
        X[[gi[int(g)] for g in ot], j] = m
    R = np.full_like(X, np.nan)
    for k in range(len(gun)):
        ok = ~np.isnan(X[k])
        cnt = ok.sum()
        if cnt >= 2:
            rk = X[k, ok].argsort().argsort()
            R[k, ok] = rk / (cnt - 1)
    return {s: (gun, R[:, j]) for j, s in enumerate(syms)}


def baglam(sym, tf, a1h, bars, d1c, btc1d, liq):
    d = bt.hazirla(bars, P0)
    c = bars[:, 4]
    ctx = dict(tf=tf, tfh=H.TF_H[tf], bars=bars, d=d, v=bars[:, 5],
               e20=bt.ema(c, 20), e50=bt.ema(c, 50), e200=bt.ema(c, 200), rsi=_rsi(c, 14),
               a1h_ot=a1h[:, 0], a1h_c=a1h[:, 4],
               coin=_gunluk_ma(d1c) if d1c is not None and len(d1c) else None,
               btc=_gunluk_ma(btc1d), liq=liq)
    return ctx


def _yapi_genel(d, i, side, P):
    """bt.detect'teki yapi taramasi, ESIKSIZ (displacement/yas/FVG/OB zorunlu degil). ICT islemlerinde ICT degerleri."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    idx = d["sh_idx"] if side == 1 else d["sl_idx"]
    px = h if side == 1 else l
    m = np.searchsorted(idx, i - P["SWING_K"])
    if m < 2:
        return None
    son_swing = idx[m - 1]; lvl = px[son_swing]
    bos_i = None
    for j in range(max(0, i - P["LOOKBACK_BOS"]), i):
        if j <= son_swing:
            continue
        if (c[j] > lvl) if side == 1 else (c[j] < lvl):
            bos_i = j; break
    if bos_i is None:
        return None
    fvg = 0.0
    for j in range(max(bos_i - 2, 2), min(bos_i + 4, i + 1)):
        if side == 1 and l[j] > h[j - 2]:
            fvg = l[j] - h[j - 2]; break
        if side == -1 and h[j] < l[j - 2]:
            fvg = l[j - 2] - h[j]; break
    ob = np.nan
    for j in range(bos_i, max(bos_i - P["OB_ARAMA"], 0), -1):
        if (side == 1 and c[j] < o[j]) or (side == -1 and c[j] > o[j]):
            ob = h[j] - l[j]; break
    return bos_i, fvg, ob


def _gun_k(ots, t):
    return np.searchsorted(ots, t - DAY, side="right") - 1      # son KAPANMIS gun


def ozellik(ctx, i, side, entry, stop):
    d = ctx["d"]; bars = ctx["bars"]; P = P0
    o, h, l, c, atr = d["o"], d["h"], d["l"], d["c"], d["atr"]
    v = ctx["v"]
    f = np.full(len(OZ), np.nan, dtype=np.float64)
    sig_close = float(bars[i, 0]) + ctx["tfh"] * HOUR
    a = atr[i]
    risk = abs(entry - stop)
    f[0] = risk / entry
    f[1] = a / c[i]
    f[2] = abs(entry - c[i]) / risk if risk > 0 else np.nan
    y = _yapi_genel(d, i, side, P)
    if y is not None:
        bos_i, fvg, ob = y
        f[3] = i - bos_i
        f[4] = abs(c[bos_i] - o[bos_i]) / atr[bos_i] if atr[bos_i] > 0 else np.nan
        f[5] = ob / a
        f[6] = fvg / a
        if bos_i >= 20:
            mv = v[bos_i - 20:bos_i].mean()
            f[8] = v[bos_i] / mv if mv > 0 else np.nan
    if i >= 20:
        mv = v[i - 20:i].mean()
        f[7] = v[i] / mv if mv > 0 else np.nan
    f[9] = (ctx["rsi"][i] - 50) * side
    if i >= 10:
        f[10] = (ctx["e20"][i] - ctx["e20"][i - 10]) / a * side
        f[11] = (ctx["e50"][i] - ctx["e50"][i - 10]) / a * side
    f[12] = (c[i] - ctx["e50"][i]) / a * side
    f[13] = (c[i] - ctx["e200"][i]) / a * side
    if ctx["coin"] is not None:
        ots, cc, mm = ctx["coin"]
        k = _gun_k(ots, sig_close)
        if k >= 49 and not np.isnan(mm[k]):
            f[14] = (1.0 if cc[k] > mm[k] else -1.0) * side
            f[15] = (cc[k] / mm[k] - 1) * side
    ots, cc, mm = ctx["btc"]
    k = _gun_k(ots, sig_close)
    if k >= 49:
        f[16] = (1.0 if cc[k] > mm[k] else -1.0) * side
        f[17] = (cc[k] / mm[k] - 1) * side
    if k >= 7:
        f[18] = (cc[k] / cc[k - 7] - 1) * side
    j = np.searchsorted(ctx["a1h_ot"], sig_close - HOUR, side="right") - 1
    c1 = ctx["a1h_c"]
    if j >= 168:
        f[19] = (c1[j] / c1[j - 168] - 1) * side
    if j >= 720:
        f[20] = (c1[j] / c1[j - 720] - 1) * side
        lr = np.diff(np.log(c1[j - 720:j + 1]))
        f[21] = lr.std() * np.sqrt(24 * 365)
    rng = h[i] - l[i]
    if rng > 0:
        up = (h[i] - max(o[i], c[i])) / rng; dn = (min(o[i], c[i]) - l[i]) / rng
        f[22], f[23] = (dn, up) if side == 1 else (up, dn)
    ms = np.searchsorted(d["sh_idx"], i - P["SWING_K"]); ml = np.searchsorted(d["sl_idx"], i - P["SWING_K"])
    if ms >= 1 and ml >= 1:
        f[24] = abs(h[d["sh_idx"][ms - 1]] - l[d["sl_idx"][ml - 1]]) / a
    f[25] = (sig_close // HOUR) % 24
    f[26] = ((sig_close // DAY) + 3) % 7
    f[27] = ctx["tfh"]
    f[28] = side
    if ctx["liq"] is not None:
        gun, rk = ctx["liq"]
        k = _gun_k(gun, sig_close)
        if k >= 0:
            f[29] = rk[k]
    return f


# =============================================================== sinyaller
def ict_sinyaller(d, P):
    """ict2._sinyaller C0_base dalinin birebir kopyasi (= bt.detect PASS)"""
    l, h = d["l"], d["h"]
    n = len(d["c"]); sig = []
    for i in range(80, n - 1):
        for side in (1, -1):
            y = ict2._yapi(d, i, side, P)
            if y is None:
                continue
            bos_i, ob = y["bos_i"], y["ob"]
            if i > bos_i + 1:
                if side == 1 and l[bos_i + 1:i].min() < ob[0]:
                    continue
                if side == -1 and h[bos_i + 1:i].max() > ob[1]:
                    continue
            pl = ict2._bolge_plan(d, i, side, ob[0], ob[1], P)
            if not pl:
                continue
            e, st, risk = pl
            sig.append(dict(i=i, side=side, entry=e, stop=st, tp=e + side * P["TP_R"] * risk,
                            pending_h=P["PENDING_H"], max_days=P["MAX_DAYS"]))
    return sig


def kural_gecer(kural, f):
    """kural: [(ozellik_idx, alt|None, ust|None), ...]  alt < x <= ust ; NaN -> median ile doldurulmus deger"""
    for j, lo, hi, med in kural:
        x = f[j]
        if np.isnan(x):
            x = med
        if lo is not None and not (x > lo):
            return False
        if hi is not None and not (x <= hi):
            return False
    return True


def _satirlar(trades, sigmap, ctx, uni, seed, sym_i, tfh, hesapla):
    meta, X = [], []
    for tr in trades:
        if tr["res"] == "cancelled":
            continue
        s = sigmap.get((tr["t"], tr["side"]))
        if s is None:
            continue
        fv = s["f"] if "f" in s else ozellik(ctx, s["i"], s["side"], s["entry"], s["stop"])
        meta.append([uni, seed, sym_i, tfh, tr["t"], tr["side"], RES[tr["res"]], tr["r"], tr["net"], tr["sp"],
                     s["i"], s["entry"], s["stop"]])
        X.append(fv)
    return meta, X


_BTC = {}


def _isci(arg):
    sym, sym_i, tf, liq, n_seed, kural = arg
    try:
        if "b" not in _BTC:
            _BTC["b"] = H.veri_1d("BTCUSDT")
        a1h = H.veri_1h(sym)
        if len(a1h) < 1000:
            return None
        bars = H.resample(a1h, tf)
        if len(bars) < 300:
            return None
        ctx = baglam(sym, tf, a1h, bars, coin_gunluk(sym, a1h), _BTC["b"], liq)
        d = ctx["d"]; tfms = H.TF_H[tf] * HOUR; tfh = H.TF_H[tf]
        S = ict_sinyaller(d, P0)
        for s in S:
            s["f"] = ozellik(ctx, s["i"], s["side"], s["entry"], s["stop"])
        say = dict(ict_sinyal=len(S))
        META, XX = [], []
        if kural is not None:           # VALID yeniden simulasyon: kural sinyal seviyesinde uygulanir
            S2 = [s for s in S if kural_gecer(kural, s["f"])]
            for uni, SS in ((0, S), (9, S2)):
                if SS:
                    T = H.sim_islem(bars, SS, tf=tf, cooldown_h=P0["COOLDOWN_H"])
                    m, x = _satirlar(T, _ilk(SS, bars, tfms), ctx, uni, 0, sym_i, tfh, True)
                    META += m; XX += x
            return dict(meta=np.array(META, float).reshape(-1, len(M)), X=np.array(XX, np.float32).reshape(-1, len(OZ)), say=say)
        if S:
            T = H.sim_islem(bars, S, tf=tf, cooldown_h=P0["COOLDOWN_H"])
            say["ict_dolan"] = sum(1 for t in T if t["res"] != "cancelled")
            m, x = _satirlar(T, _ilk(S, bars, tfms), ctx, 0, 0, sym_i, tfh, True)
            META += m; XX += x
        n_sig = len(S); i_list = [s["i"] for s in S]
        if n_sig >= 3:
            for mod, uni in (("zaman", 1), ("rastgele", 2)):
                for sd in range(n_seed):
                    seed = int(hashlib.md5(f"{sym}|{tf}|{mod}|{sd}".encode()).hexdigest()[:8], 16)
                    ks = ict2._kontrol_sinyaller(d, bars, tf, n_sig, seed, P0, mod, i_list)
                    if not ks:
                        continue
                    T = H.sim_islem(bars, ks, tf=tf, cooldown_h=P0["COOLDOWN_H"])
                    m, x = _satirlar(T, _ilk(ks, bars, tfms), ctx, uni, sd, sym_i, tfh, True)
                    META += m; XX += x
        return dict(meta=np.array(META, float).reshape(-1, len(M)), X=np.array(XX, np.float32).reshape(-1, len(OZ)), say=say)
    except Exception as ex:
        import traceback
        return dict(hata=f"{sym} {tf}: {ex} {traceback.format_exc()[-400:]}")


def _ilk(SS, bars, tfms):
    """(sig_close, side) -> sim_islem'in aldigi ILK sinyal (liste i'ye gore sirali; esitlikte ilk gelen)"""
    sm = {}
    for s in SS:
        k = (int(bars[s["i"], 0] + tfms), s["side"])
        if k not in sm:
            sm[k] = s
    return sm


# =============================================================== asama: uret
def _liq_hazirla(syms):
    qv = {}
    for s in syms:
        a = H.veri_1h(s)
        if len(a) >= 1000:
            qv[s] = gunluk_qv(a)
    return qv, likidite_panel(qv)


def uret(syms, onek, isci=4, n_seed=N_SEED):
    os.makedirs(ARA, exist_ok=True)
    qv, liq = _liq_hazirla(syms)
    ozet = {}
    for tf in TFS:
        yol = os.path.join(ARA, f"{onek}_veri_{tf}.npz")
        if os.path.exists(yol):
            print(f"[{tf}] ara sonuc var, atlandi: {yol}", flush=True)
            continue
        t0 = time.time()
        args = [(s, k, tf, liq.get(s), n_seed, None) for k, s in enumerate(syms)]
        MM, XX, hatalar, sig, dol = [], [], [], 0, 0
        with ProcessPoolExecutor(max_workers=isci) as ex:
            for r in ex.map(_isci, args, chunksize=1):
                if r is None:
                    continue
                if "hata" in r:
                    hatalar.append(r["hata"]); continue
                MM.append(r["meta"]); XX.append(r["X"])
                sig += r["say"].get("ict_sinyal", 0); dol += r["say"].get("ict_dolan", 0)
        meta = np.concatenate(MM) if MM else np.zeros((0, len(M)))
        X = np.concatenate(XX) if XX else np.zeros((0, len(OZ)), np.float32)
        np.savez_compressed(yol, meta=meta, X=X)
        ozet[tf] = dict(ict_sinyal=sig, ict_dolan=dol, satir=int(len(meta)), sure_s=round(time.time() - t0), hata=len(hatalar))
        print(f"[{tf}] sinyal={sig} dolan={dol} satir={len(meta)} {time.time()-t0:.0f}s hata={len(hatalar)}", flush=True)
        if hatalar:
            print("  HATA ornek:", hatalar[:2], flush=True)
        del MM, XX, meta, X
    return ozet


_YUK = {}


def yukle(onek):
    if onek in _YUK:
        return _YUK[onek]
    MM, XX = [], []
    for tf in TFS:
        yol = os.path.join(ARA, f"{onek}_veri_{tf}.npz")
        if os.path.exists(yol):
            z = np.load(yol)
            MM.append(z["meta"]); XX.append(z["X"])
    _YUK.clear()
    _YUK[onek] = (np.concatenate(MM), np.concatenate(XX).astype(np.float64))
    return _YUK[onek]


# =============================================================== asama: esdegerlik ve nedensellik
def esdeger(syms):
    """kendi ICT sinyallerimiz == bt.detect PASS == ict2._sinyaller C0_base"""
    sonuc = dict(kontrol=0, uyusmaz_bt=0, uyusmaz_ict2=0)
    for s in syms:
        a1h = H.veri_1h(s)
        for tf in TFS:
            bars = H.resample(a1h, tf)
            if len(bars) < 300:
                continue
            d = bt.hazirla(bars, P0)
            ben = {(x["i"], x["side"], round(x["entry"], 10), round(x["stop"], 10)) for x in ict_sinyaller(d, P0)}
            btk = set()
            for i in range(80, d["n"] - 1):
                for side in (1, -1):
                    pl, sb = bt.detect(d, i, side, P0)
                    if pl:
                        btk.add((i, side, round(pl["entry"], 10), round(pl["stop"], 10)))
            ic = {(x["i"], x["side"], round(x["entry"], 10), round(x["stop"], 10))
                  for x in ict2._sinyaller(s, tf, bars, d, P0)["C0_base"]}
            sonuc["kontrol"] += len(ben)
            sonuc["uyusmaz_bt"] += len(ben ^ btk)
            sonuc["uyusmaz_ict2"] += len(ben ^ ic)
    return sonuc


def nedensellik(onek, syms, n_test=60, seed=11):
    meta, X = yukle(onek)
    rng = np.random.default_rng(seed)
    secim = list(rng.choice(np.flatnonzero(meta[:, M["uni"]] == 0), n_test // 2, replace=False)) + \
        list(rng.choice(np.flatnonzero(meta[:, M["uni"]] > 0), n_test - n_test // 2, replace=False))
    qv_full, _ = _liq_hazirla(syms)
    btc_full = H.veri_1d("BTCUSDT")
    tfmap = {v: k for k, v in H.TF_H.items()}
    uyusan, kotu, detay = 0, 0, []
    for r in secim:
        mr = meta[r]; sym = syms[int(mr[M["sym"]])]; tf = tfmap[int(mr[M["tfh"]])]
        i = int(mr[M["i"]]); side = int(mr[M["side"]])
        a1h = H.veri_1h(sym); bars = H.resample(a1h, tf)
        sig_close = float(bars[i, 0]) + H.TF_H[tf] * HOUR
        # KESILMIS veri: sinyal kapanisindan sonra hicbir sey yok
        a1h_k = a1h[a1h[:, 0] + HOUR <= sig_close]
        bars_k = H.resample(a1h_k, tf)
        d1c = coin_gunluk(sym, a1h); d1c_k = d1c[d1c[:, 0] + DAY <= sig_close]
        btc_k = btc_full[btc_full[:, 0] + DAY <= sig_close]
        qv_k = {s: (ot[ot + DAY <= sig_close], q[ot + DAY <= sig_close]) for s, (ot, q) in qv_full.items()}
        liq_k = likidite_panel(qv_k).get(sym)
        ik = len(bars_k) - 1
        assert bars_k[ik, 0] == bars[i, 0], "kesilmis veride sinyal bari son bar degil"
        ctx_k = baglam(sym, tf, a1h_k, bars_k, d1c_k, btc_k, liq_k)
        fk = ozellik(ctx_k, ik, side, mr[M["entry"]], mr[M["stop"]])
        ff = X[r]
        esit = np.isclose(ff, fk.astype(np.float32).astype(np.float64), rtol=1e-4, atol=1e-6, equal_nan=True)
        if esit.all():
            uyusan += 1
        else:
            kotu += 1
            detay.append(dict(sym=sym, tf=tf, i=i, farkli=[OZ[j] for j in np.flatnonzero(~esit)],
                              tam=[float(ff[j]) for j in np.flatnonzero(~esit)], kesik=[float(fk[j]) for j in np.flatnonzero(~esit)]))
    return dict(test=len(secim), ayni=uyusan, farkli=kotu, detay=detay[:10])


# =============================================================== madencilik
def _w(net):
    return np.clip(net, *WINS)


def _top1(net):
    if len(net) < 20 or net.sum() == 0:
        return None
    xs = np.sort(net)[::-1]; k = max(1, int(len(net) * 0.01))
    return float(xs[:k].sum() / abs(net.sum()))


def _katlar(t, sinir):
    return np.searchsorted(sinir, t, side="right")


def _yaprak_kural(tree, Xtr, yap_tr, wtr, min_n=100):
    """en iyi yaprak (egitim katlarinda winsorize ort) ve yol esikleri"""
    best, bl = -1e9, None
    for lf in np.unique(yap_tr):
        mk = yap_tr == lf
        if mk.sum() >= min_n and wtr[mk].mean() > best:
            best, bl = wtr[mk].mean(), lf
    return bl


def _yol(tree, yaprak):
    t = tree.tree_
    parent = {}
    for nd in range(t.node_count):
        for ch, yon in ((t.children_left[nd], "le"), (t.children_right[nd], "gt")):
            if ch != -1:
                parent[ch] = (nd, yon)
    kos = []; nd = yaprak
    while nd in parent:
        p, yon = parent[nd]
        kos.append((int(t.feature[p]), yon, float(t.threshold[p])))
        nd = p
    return kos[::-1]


def _kosul_sade(kos):
    """[(j,'le'/'gt',thr)] -> {j: [alt, ust]}"""
    d = {}
    for j, yon, thr in kos:
        lo, hi = d.get(j, [None, None])
        if yon == "le":
            hi = thr if hi is None else min(hi, thr)
        else:
            lo = thr if lo is None else max(lo, thr)
        d[j] = [lo, hi]
    return d


def _kosul_maske(X, kd):
    mk = np.ones(len(X), bool)
    for j, (lo, hi) in kd.items():
        if lo is not None:
            mk &= X[:, j] > lo
        if hi is not None:
            mk &= X[:, j] <= hi
    return mk


KONFIGLER = ["logit_top30", "logit_top50", "agac2_top30", "agac2_top50", "agac3_top30", "agac3_top50",
             "K_agac2_yaprak", "K_agac3_yaprak", "K_logit_vekil", "K_tek_ozellik"]
KURAL_KONFIG = [k for k in KONFIGLER if k.startswith("K_")]


def kurallar_egit(Xtr, ytr, wtr):
    """egitim satirlarindan tum modeller + basit kurallar. Return dict ad -> (skor_fn | kosul dict)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    Xi = np.where(np.isnan(Xtr), med, Xtr)
    mu = Xi.mean(0); sd = Xi.std(0); sd[sd == 0] = 1
    out = dict(med=med)
    if len(np.unique(ytr)) < 2:
        return None
    lg = LogisticRegression(C=1.0, max_iter=1000).fit((Xi - mu) / sd, ytr)
    out["logit"] = (lambda X, lg=lg, mu=mu, sd=sd: lg.decision_function((X - mu) / sd))
    s_lg = out["logit"](Xi)
    for dep in (2, 3):
        tr = DecisionTreeClassifier(max_depth=dep, min_samples_leaf=100, random_state=0).fit(Xi, ytr)
        out[f"agac{dep}"] = (lambda X, tr=tr: tr.predict_proba(X)[:, -1])
        bl = _yaprak_kural(tr, Xi, tr.apply(Xi), wtr)
        out[f"K_agac{dep}_yaprak"] = _kosul_sade(_yol(tr, bl)) if bl is not None else {}
    rg = DecisionTreeRegressor(max_depth=2, min_samples_leaf=100, random_state=0).fit(Xi, s_lg)
    bl = _yaprak_kural(rg, Xi, rg.apply(Xi), wtr)
    out["K_logit_vekil"] = _kosul_sade(_yol(rg, bl)) if bl is not None else {}
    # tek ozellik: 5'li dilim, en iyi (ozellik, dilim) egitimde winsorize ort
    best, bk = -1e9, {}
    for j in range(Xi.shape[1]):
        ed = np.unique(np.quantile(Xi[:, j], [0.2, 0.4, 0.6, 0.8]))
        b = np.searchsorted(ed, Xi[:, j], side="left")      # x<=ed[0] -> 0
        for k in range(len(ed) + 1):
            mk = b == k
            if mk.sum() >= 100 and wtr[mk].mean() > best:
                best = wtr[mk].mean()
                bk = {j: [None if k == 0 else float(ed[k - 1]), None if k == len(ed) else float(ed[k])]}
    out["K_tek_ozellik"] = bk
    out["s_tr"] = dict(logit=s_lg, agac2=out["agac2"](Xi), agac3=out["agac3"](Xi))
    return out


def kurallari_uygula(mod, X):
    Xi = np.where(np.isnan(X), mod["med"], X)
    mk = {}
    for base in ("logit", "agac2", "agac3"):
        s = mod[base](Xi); st = mod["s_tr"][base]
        for q, ad in ((0.70, "top30"), (0.50, "top50")):
            mk[f"{base}_{ad}"] = s >= np.quantile(st, q)
    for k in KURAL_KONFIG:
        mk[k] = _kosul_maske(Xi, mod[k])
    return mk


def pipeline(X, y, net, t, sinir):
    """kat-disi (OOF) filtre sonuclari. Return dict konfig -> metrikler"""
    w = _w(net); kat = _katlar(t, sinir)
    MASK = {k: np.zeros(len(y), bool) for k in KONFIGLER}
    gecerli = np.zeros(len(y), bool)
    for k in range(4):
        te = kat == k
        if te.sum() == 0:
            continue
        lo, hi = t[te].min(), t[te].max()
        tr = (kat != k) & ((t < lo - GAP) | (t > hi + GAP))
        if tr.sum() < 250:
            continue
        mod = kurallar_egit(X[tr], y[tr], w[tr])
        if mod is None:
            continue
        mk = kurallari_uygula(mod, X[te])
        idx = np.flatnonzero(te); gecerli[idx] = True
        for ad in KONFIGLER:
            MASK[ad][idx] = mk[ad]
    base = float(w[gecerli].mean()) if gecerli.any() else np.nan
    res = dict(_base=dict(n=int(gecerli.sum()), wins=round(base, 4), net=round(float(net[gecerli].mean()), 4) if gecerli.any() else None,
                          win=round(float(y[gecerli].mean()), 4) if gecerli.any() else None))
    for ad in KONFIGLER:
        mk = MASK[ad] & gecerli
        n = int(mk.sum())
        katw = [round(float(w[mk & (kat == k)].mean()), 4) if (mk & (kat == k)).sum() else None for k in range(4)]
        fw = float(w[mk].mean()) if n else np.nan
        res[ad] = dict(n=n, wins=round(fw, 4) if n else None, kazanc=round(fw - base, 4) if n else None,
                       net=round(float(net[mk].mean()), 4) if n else None, win=round(float(y[mk].mean()), 4) if n else None,
                       kat_wins=katw, kat_pozitif=int(sum(1 for v in katw if v is not None and v > 0)),
                       top1_pay=(round(_top1(net[mk]), 3) if n >= 20 and _top1(net[mk]) is not None else None))
    return res


def tek_degiskenli(X, y, net, t, sinir):
    """her ozellik: tum TRAIN'de 5'li dilim tablosu + kat-disi tutarlilik (dilim egitim katlarinda secilir)"""
    w = _w(net); kat = _katlar(t, sinir); base = w.mean()
    tab = {}
    for j, ad in enumerate(OZ):
        x = X[:, j]; ok = ~np.isnan(x)
        if ok.sum() < 500:
            tab[ad] = dict(n=int(ok.sum()), not_="yetersiz")
            continue
        ed = np.unique(np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8]))
        b = np.where(ok, np.searchsorted(ed, np.nan_to_num(x), side="left"), -1)
        dil = []
        for k in range(len(ed) + 1):
            mk = b == k
            dil.append(dict(alt=None if k == 0 else round(float(ed[k - 1]), 5), ust=None if k == len(ed) else round(float(ed[k]), 5),
                            n=int(mk.sum()), wins=round(float(w[mk].mean()), 4) if mk.any() else None,
                            win=round(float(y[mk].mean()), 3) if mk.any() else None))
        en = max(range(len(dil)), key=lambda k: -1e9 if dil[k]["n"] < 100 else dil[k]["wins"])
        # kat-disi: dilim egitim katlarinda secilir, test katinda olculur
        oof_kaz, oof_n, poz = [], 0, 0
        sw, sn = 0.0, 0
        for kk in range(4):
            te = (kat == kk) & ok
            if te.sum() == 0:
                continue
            lo, hi = t[te].min(), t[te].max()
            tr = (kat != kk) & ok & ((t < lo - GAP) | (t > hi + GAP))
            if tr.sum() < 250:
                continue
            e2 = np.unique(np.quantile(x[tr], [0.2, 0.4, 0.6, 0.8]))
            btr = np.searchsorted(e2, x[tr], side="left"); bte = np.searchsorted(e2, x[te], side="left")
            cand = [(w[tr][btr == k].mean(), k) for k in range(len(e2) + 1) if (btr == k).sum() >= 100]
            if not cand:
                continue
            kb = max(cand)[1]
            mk = bte == kb
            if mk.sum():
                g = w[te][mk].mean() - w[te].mean()
                oof_kaz.append(round(float(g), 4)); poz += int(w[te][mk].mean() > 0)
                sw += w[te][mk].sum(); sn += int(mk.sum())
        tab[ad] = dict(dilimler=dil, en_iyi_dilim=en, en_iyi_wins=dil[en]["wins"], en_iyi_n=dil[en]["n"],
                       en_iyi_kazanc=round(float(dil[en]["wins"] - base), 4) if dil[en]["wins"] is not None else None,
                       oof_kat_kazanc=oof_kaz, oof_wins=round(sw / sn, 4) if sn else None, oof_n=sn, oof_kat_pozitif_wins=poz)
    return tab


def betimleyici(X, y):
    """kazanan(TP) vs kaybeden: ortalama farki ve Cohen d"""
    out = {}
    for j, ad in enumerate(OZ):
        a = X[y == 1, j]; b = X[y == 0, j]
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        out[ad] = dict(kazanan_ort=float(a.mean()), kaybeden_ort=float(b.mean()), fark=float(a.mean() - b.mean()),
                       d=float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0, n_kaz=int(len(a)), n_kay=int(len(b)))
    return out


def _evren(meta, X, uni, seed=None, bolum="train"):
    a, b = H.BOLUM_1H[bolum]
    mk = (meta[:, M["uni"]] == uni) & (meta[:, M["t"]] >= a) & (meta[:, M["t"]] < b)
    if seed is not None:
        mk &= meta[:, M["seed"]] == seed
    if uni > 0 and seed is not None:
        # ESIT ORNEKLEM: plasebo seed'i, ayni bolumdeki ICT dolan islem sayisina (dilim bazinda) alt-orneklenir
        ict_mk = (meta[:, M["uni"]] == 0) & (meta[:, M["t"]] >= a) & (meta[:, M["t"]] < b)
        sec = np.zeros(len(meta), bool)
        for tfh in np.unique(meta[mk, M["tfh"]]):
            havuz = np.flatnonzero(mk & (meta[:, M["tfh"]] == tfh))
            hedef = int((ict_mk & (meta[:, M["tfh"]] == tfh)).sum())
            rng = np.random.default_rng(int(7919 * seed + 31 * uni + tfh) + (0 if bolum == "train" else 100000))
            sec[rng.choice(havuz, min(hedef, len(havuz)), replace=False)] = True
        mk = sec
    mm = meta[mk]; xx = X[mk]
    o = np.argsort(mm[:, M["t"]], kind="stable")
    mm, xx = mm[o], xx[o]
    return xx, (mm[:, M["res"]] == 1).astype(int), mm[:, M["net"]], mm[:, M["t"]], mm


def _rap(mm, maske=None):
    if maske is not None:
        mm = mm[maske]
    T = [dict(t=float(r[M["t"]]), res="x", net=float(r[M["net"]]), r=float(r[M["r"]])) for r in mm]
    rp = H.rapor(T)
    if len(mm) >= 20:
        rp["wins3"] = round(float(_w(mm[:, M["net"]]).mean()), 4)
        rp["top1_pay"] = round(_top1(mm[:, M["net"]]), 3) if _top1(mm[:, M["net"]]) is not None else None
    return rp


def _pool_seed(arg):
    onek, uni, sd, sinir = arg
    meta, X = yukle(onek)
    Xs, ys, ns, ts, _ = _evren(meta, X, uni, sd)
    del meta, X
    return uni, sd, pipeline(Xs, ys, ns, ts, sinir), betimleyici(Xs, ys)


def _pool_perm(arg):
    onek, p, sinir = arg
    meta, X = yukle(onek)
    Xi, yi, ni, ti, _ = _evren(meta, X, 0)
    del meta, X
    rng = np.random.default_rng(1000 + p)
    kat = _katlar(ti, sinir)
    perm = np.arange(len(yi))
    for k in range(4):
        idx = np.flatnonzero(kat == k); perm[idx] = rng.permutation(idx)
    return pipeline(Xi, yi[perm], ni[perm], ti, sinir)


def madencilik(onek, isci=4, n_seed=N_SEED, n_perm=N_PERM):
    meta, X = yukle(onek)
    Xi, yi, ni, ti, mi = _evren(meta, X, 0)
    sinir = np.quantile(ti, [0.25, 0.5, 0.75])
    print(f"ICT TRAIN dolan islem n={len(yi)} win={yi.mean():.3f} net={ni.mean():+.4f} wins={_w(ni).mean():+.4f}", flush=True)
    ict = pipeline(Xi, yi, ni, ti, sinir)
    uni_tab = tek_degiskenli(Xi, yi, ni, ti, sinir)
    bet_ict = betimleyici(Xi, yi)
    # evren ozetleri
    evren_ozet = {"ICT": _rap(mi)}
    for uni, ad in ((1, "P1_ayni_bar_rastgele_yon"), (2, "P2_rastgele_bar_yon")):
        _, yy, nn, _, mm = _evren(meta, X, uni)
        sn = [float(_w(nn[mm[:, M["seed"]] == s]).mean()) for s in range(n_seed) if (mm[:, M["seed"]] == s).any()]
        evren_ozet[ad] = dict(havuz_n=int(len(nn)), seed_basi_n_ort=round(len(nn) / max(1, n_seed), 1),
                              net=round(float(nn.mean()), 4), wins3=round(float(_w(nn).mean()), 4),
                              win=round(float(yy.mean()), 4), seed_wins3_p5_50_95=[round(float(np.percentile(sn, q)), 4) for q in (5, 50, 95)])
    del meta, X
    t0 = time.time()
    PL = {1: {}, 2: {}}; BET = {1: {}, 2: {}}
    with ProcessPoolExecutor(max_workers=isci) as ex:
        for uni, sd, r, b in ex.map(_pool_seed, [(onek, u, s, sinir) for u in (1, 2) for s in range(n_seed)]):
            PL[uni][sd] = r; BET[uni][sd] = b
        print(f"plasebo pipeline bitti {time.time()-t0:.0f}s", flush=True)
        PERM = list(ex.map(_pool_perm, [(onek, p, sinir) for p in range(n_perm)]))
    print(f"permutasyon bitti {time.time()-t0:.0f}s", flush=True)
    return dict(sinir=sinir, ict=ict, uni_tab=uni_tab, bet_ict=bet_ict, PL=PL, BET=BET, PERM=PERM, evren_ozet=evren_ozet,
                ict_n=len(yi))


def _dagilim(PLu, ad):
    v = [PLu[s][ad]["kazanc"] for s in PLu if PLu[s][ad]["kazanc"] is not None]
    return v


def degerlendir(R):
    tablo = {}; adaylar = []
    for ad in KONFIGLER:
        ic = R["ict"][ad]
        p1 = _dagilim(R["PL"][1], ad); p2 = _dagilim(R["PL"][2], ad)
        pv = [pp[ad]["kazanc"] for pp in R["PERM"] if pp[ad]["kazanc"] is not None]
        kaz = ic["kazanc"]
        p1_95 = float(np.percentile(p1, 95)) if p1 else None
        p2_95 = float(np.percentile(p2, 95)) if p2 else None
        perm_p = (1 + sum(1 for v in pv if kaz is not None and v >= kaz)) / (1 + len(pv)) if kaz is not None else None
        sart = dict(
            i_plasebo=bool(kaz is not None and p1_95 is not None and p2_95 is not None and kaz > p1_95 and kaz > p2_95),
            ii_perm=bool(perm_p is not None and perm_p < 0.01),
            iii_pozitif_3kat=bool(ic["wins"] is not None and ic["wins"] > 0 and ic["kat_pozitif"] >= 3),
            iv_n200=bool(ic["n"] >= 200),
            v_top1=bool(ic["top1_pay"] is not None and ic["top1_pay"] < 0.5),
            vi_basit=ad.startswith("K_"))
        tablo[ad] = dict(ict=ic, P1_kazanc_p5_50_95=[round(float(np.percentile(p1, q)), 4) for q in (5, 50, 95)] if p1 else None,
                         P2_kazanc_p5_50_95=[round(float(np.percentile(p2, q)), 4) for q in (5, 50, 95)] if p2 else None,
                         P1_filtreli_wins_medyan=round(float(np.median([R["PL"][1][s][ad]["wins"] for s in R["PL"][1] if R["PL"][1][s][ad]["wins"] is not None])), 4),
                         P2_filtreli_wins_medyan=round(float(np.median([R["PL"][2][s][ad]["wins"] for s in R["PL"][2] if R["PL"][2][s][ad]["wins"] is not None])), 4),
                         ict_kazanc_P1_yuzdelik=round(float(np.mean([v < kaz for v in p1])) * 100, 1) if (p1 and kaz is not None) else None,
                         ict_kazanc_P2_yuzdelik=round(float(np.mean([v < kaz for v in p2])) * 100, 1) if (p2 and kaz is not None) else None,
                         perm_kazanc_p95=round(float(np.percentile(pv, 95)), 4) if pv else None, perm_p=round(perm_p, 4) if perm_p else None,
                         sartlar=sart, gecti=all(sart.values()))
        if all(sart.values()):
            adaylar.append(ad)
    return tablo, adaylar


def betimleyici_tablo(R, n_seed):
    bi = R["bet_ict"]
    # P1/P2 havuz betimleyici (tum seedler ayri ayri -> d dagilimi)
    sirali = sorted(bi, key=lambda k: -abs(bi[k]["d"]))
    top = sirali[:10]
    p1_top = sorted({k for s in R["BET"][1] for k in R["BET"][1][s]},
                    key=lambda k: -abs(np.median([R["BET"][1][s][k]["d"] for s in R["BET"][1] if k in R["BET"][1][s]])))[:10]
    satir = []
    for k in top:
        d1 = [R["BET"][1][s][k]["d"] for s in R["BET"][1] if k in R["BET"][1][s]]
        d2 = [R["BET"][2][s][k]["d"] for s in R["BET"][2] if k in R["BET"][2][s]]
        f1 = [R["BET"][1][s][k]["fark"] for s in R["BET"][1] if k in R["BET"][1][s]]
        dd = bi[k]["d"]
        satir.append(dict(ozellik=k, ICT_kazanan_ort=round(bi[k]["kazanan_ort"], 4), ICT_kaybeden_ort=round(bi[k]["kaybeden_ort"], 4),
                          ICT_fark=round(bi[k]["fark"], 4), ICT_d=round(dd, 3), ICT_n=[bi[k]["n_kaz"], bi[k]["n_kay"]],
                          P1_fark_medyan=round(float(np.median(f1)), 4) if f1 else None,
                          P1_d_medyan=round(float(np.median(d1)), 3) if d1 else None,
                          P1_d_p5_p95=[round(float(np.percentile(d1, 5)), 3), round(float(np.percentile(d1, 95)), 3)] if d1 else None,
                          P2_d_medyan=round(float(np.median(d2)), 3) if d2 else None,
                          P2_d_p5_p95=[round(float(np.percentile(d2, 5)), 3), round(float(np.percentile(d2, 95)), 3)] if d2 else None,
                          ICT_d_P1_bandi_disinda=bool(d1 and not (np.percentile(d1, 5) <= dd <= np.percentile(d1, 95))),
                          ayni_isaret_P1=bool(d1 and np.sign(np.median(d1)) == np.sign(dd))))
    p1_satir = []
    for k in p1_top:
        d1 = [R["BET"][1][s][k]["d"] for s in R["BET"][1] if k in R["BET"][1][s]]
        p1_satir.append(dict(ozellik=k, P1_d_medyan=round(float(np.median(d1)), 3),
                             P1_d_p5_p95=[round(float(np.percentile(d1, 5)), 3), round(float(np.percentile(d1, 95)), 3)],
                             ICT_d=round(bi[k]["d"], 3) if k in bi else None))
    return dict(aciklama=("TRAIN dolan islemler. d = Cohen d (kazanan TP - kaybeden, havuzlanmis sd). P1 = ICT'nin ayni barlari + "
                          "rastgele yon (her seed ayri tablo, ICT ile ayni orneklem buyuklugu); P2 = rastgele bar+yon. "
                          "ICT_d_P1_bandi_disinda=True ise ICT'deki fark plasebo seedlerinin %5-%95 bandinin disinda. " + OZ_ACIKLAMA),
                ICT_en_farkli_10=satir, P1_en_farkli_10=p1_satir)


# =============================================================== VALID
def valid_calistir(onek, syms, adaylar, R, isci=4, n_seed=N_SEED):
    meta, X = yukle(onek)
    Xtr, ytr, ntr, ttr, _ = _evren(meta, X, 0, bolum="train")
    Xva, yva, nva, tva, mva = _evren(meta, X, 0, bolum="valid")
    mod = kurallar_egit(Xtr, ytr, _w(ntr))
    out = {}
    for ad in adaylar:
        kd = mod[ad]
        mk = kurallari_uygula(mod, Xva)[ad]
        base = float(_w(nva).mean())
        ict_v = dict(kural={OZ[j]: v for j, v in kd.items()}, filtresiz=_rap(mva), filtreli=_rap(mva, mk),
                     kazanc=round(float(_w(nva[mk]).mean() - base), 4) if mk.any() else None)
        pk = {1: [], 2: []}
        for uni in (1, 2):
            for sd in range(n_seed):
                Xs, ys, ns, ts, _ = _evren(meta, X, uni, sd, "train")
                Xv, yv, nv, tv, _ = _evren(meta, X, uni, sd, "valid")
                if len(ys) < 300 or len(yv) < 50:
                    continue
                m2 = kurallar_egit(Xs, ys, _w(ns))
                if m2 is None:
                    continue
                mm2 = kurallari_uygula(m2, Xv)[ad]
                if mm2.sum():
                    pk[uni].append(float(_w(nv[mm2]).mean() - _w(nv).mean()))
        ict_v["P1_kazanc_p5_50_95"] = [round(float(np.percentile(pk[1], q)), 4) for q in (5, 50, 95)] if pk[1] else None
        ict_v["P2_kazanc_p5_50_95"] = [round(float(np.percentile(pk[2], q)), 4) for q in (5, 50, 95)] if pk[2] else None
        # yeniden simulasyon: kural sinyal seviyesinde (kilit/cooldown etkisi dahil)
        kural = [(j, lo, hi, float(mod["med"][j])) for j, (lo, hi) in kd.items()]
        qv, liq = _liq_hazirla(syms)
        MM = []
        for tf in TFS:
            with ProcessPoolExecutor(max_workers=isci) as ex:
                for r in ex.map(_isci, [(s, k, tf, liq.get(s), 0, kural) for k, s in enumerate(syms)]):
                    if r and "meta" in r:
                        MM.append(r["meta"])
        mm = np.concatenate(MM)
        a, b = H.BOLUM_1H["valid"]
        for uni, nm in ((0, "yeniden_sim_filtresiz"), (9, "yeniden_sim_filtreli")):
            s = mm[(mm[:, M["uni"]] == uni) & (mm[:, M["t"]] >= a) & (mm[:, M["t"]] < b)]
            ict_v[nm] = _rap(s)
        out[ad] = ict_v
        H.kaydet(AILE, f"VALID|{ad}", dict(kural=ict_v["kural"]), "valid", ict_v, "TEK SEFERLIK valid, ayni plasebo karsilastirmasi")
    return out


def _sabit_dilim_oof(x, w, net, t, sinir, dilim=4):
    """SABIT kural: ozelligin 5'li dilimlerinden 'dilim' (egitim katlarinda hesaplanan sinirlarla), kat-disi"""
    kat = _katlar(t, sinir); ok = ~np.isnan(x)
    kaz, poz, sw, sn, snet = [], 0, 0.0, 0, 0.0
    for kk in range(4):
        te = (kat == kk) & ok
        if te.sum() == 0:
            continue
        lo, hi = t[te].min(), t[te].max()
        tr = (kat != kk) & ok & ((t < lo - GAP) | (t > hi + GAP))
        if tr.sum() < 100:          # sabit kural: sadece dilim sinirlari tahmin edilir
            continue
        e2 = np.quantile(x[tr], [0.2, 0.4, 0.6, 0.8])
        mk = np.searchsorted(e2, x[te], side="left") == dilim
        if mk.sum():
            kaz.append(float(w[te][mk].mean() - w[te].mean())); poz += int(w[te][mk].mean() > 0)
            sw += w[te][mk].sum(); sn += int(mk.sum()); snet += net[te][mk].sum()
    return dict(kat_kazanc=[round(v, 4) for v in kaz], kazanc_ort=round(float(np.mean(kaz)), 4) if kaz else None,
                kat_hepsi_kazancli=bool(len(kaz) == 4 and min(kaz) > 0), wins=round(sw / sn, 4) if sn else None,
                net=round(snet / sn, 4) if sn else None, n=sn, kat_pozitif_wins=poz)


def sonradan(onek, oz_ad="fvg_atr", dilim=4, n_seed=N_SEED, n_perm=N_PERM):
    """SONRADAN (post-hoc, aday DEGIL): ICT tek degiskenli tabloda goze carpan sabit kurali plasebo + permutasyonla sina"""
    meta, X = yukle(onek); j = OZ.index(oz_ad)
    Xi, yi, ni, ti, _ = _evren(meta, X, 0)
    sinir = np.quantile(ti, [0.25, 0.5, 0.75])
    ict = _sabit_dilim_oof(Xi[:, j], _w(ni), ni, ti, sinir, dilim)
    pl = {}
    for uni in (1, 2):
        v = []
        for sd in range(n_seed):
            Xs, ys, ns, ts, _ = _evren(meta, X, uni, sd)
            r = _sabit_dilim_oof(Xs[:, j], _w(ns), ns, ts, sinir, dilim)
            if r["kazanc_ort"] is not None:
                v.append(r)
        g = [r["kazanc_ort"] for r in v]
        if not g:
            pl[f"P{uni}"] = dict(seed=0, not_="yeterli ozellik degeri yok (yapi yok -> NaN)")
            continue
        pl[f"P{uni}"] = dict(seed=len(v), kazanc_p5_50_95=[round(float(np.percentile(g, q)), 4) for q in (5, 50, 95)],
                             kat_hepsi_kazancli_orani=round(float(np.mean([r["kat_hepsi_kazancli"] for r in v])), 3),
                             wins_medyan=round(float(np.median([r["wins"] for r in v])), 4))
    kat = _katlar(ti, sinir); pg = []
    for p in range(n_perm):
        rng = np.random.default_rng(5000 + p); perm = np.arange(len(yi))
        for k in range(4):
            idx = np.flatnonzero(kat == k); perm[idx] = rng.permutation(idx)
        r = _sabit_dilim_oof(Xi[:, j], _w(ni[perm]), ni[perm], ti, sinir, dilim)
        if r["kazanc_ort"] is not None:
            pg.append(r["kazanc_ort"])
    perm_p = (1 + sum(1 for g in pg if g >= ict["kazanc_ort"])) / (1 + len(pg))
    return dict(kural=f"{oz_ad} dilim {dilim}/4 (0=en alt %20, 4=en ust %20; sinirlar egitim katlarindan)", not_="SONRADAN secildi (30 ozellikten en iyi gorunen); coklu test duzeltmesi YOK -> aday sayilmaz",
                ICT=ict, plasebo=pl, perm_kazanc_p95=round(float(np.percentile(pg, 95)), 4), perm_p=round(perm_p, 4),
                coklu_test_bonferroni_30=round(min(1.0, perm_p * 30), 4))


# =============================================================== sonuc json
def rapor_yaz(onek, ARAJ, ogrenilen, karar, oneri):
    """ara json -> araclar/lab/sonuc_ict_ozellik.json (sablon formati + betimleyici + karar)"""
    md = ARAJ["madencilik"]; tab = md["tablo"]
    sirali = sorted(KURAL_KONFIG, key=lambda k: -(tab[k]["ict"]["kazanc"] if tab[k]["ict"]["kazanc"] is not None else -9))
    en3 = []
    for k in sirali[:3]:
        x = tab[k]
        en3.append(dict(ad=k, params=dict(kat=4, bosluk_gun=7, min_yaprak=100, hedef="kazandi=TP", bolum="train OOF"),
                        train=dict(ict_filtreli=x["ict"], ict_filtresiz=md["ict_base"],
                                   P1_kazanc_p5_50_95=x["P1_kazanc_p5_50_95"], P2_kazanc_p5_50_95=x["P2_kazanc_p5_50_95"],
                                   perm_p=x["perm_p"], sartlar=x["sartlar"]),
                        valid=(ARAJ.get("valid") or {}).get(k),
                        komsular="ayni pipeline'in model-skoru filtreleri (logit/agac top30/top50) tabloda: madencilik.tablo"))
    dusen = []
    for k in KONFIGLER:
        x = tab[k]
        if not x["gecti"]:
            olmayan = [s for s, v in x["sartlar"].items() if not v]
            dusen.append(f"{k}: ICT OOF kazanc={x['ict']['kazanc']} wins={x['ict']['wins']} n={x['ict']['n']} "
                         f"katpoz={x['ict']['kat_pozitif']}/4, P1 p95={x['P1_kazanc_p5_50_95'][2] if x['P1_kazanc_p5_50_95'] else None}, "
                         f"P2 p95={x['P2_kazanc_p5_50_95'][2] if x['P2_kazanc_p5_50_95'] else None}, perm_p={x['perm_p']} -> kalan sart: {olmayan}")
    out = dict(aile=AILE, denenen=len(KONFIGLER) + len(OZ),
               not_=("Denenen = 10 filtre konfigurasyonu (6 model-skoru + 4 basit kural) + 30 tek degiskenli ozellik; "
                     "her biri ICT + P1x20 + P2x20 seed + 100 permutasyon. Parametre taramasi yok (bt.P0 sabit)."),
               en_iyi_3=en3, dusenler=dusen, ogrenilen=ogrenilen, oneri=oneri,
               betimleyici=md["betimleyici"], karar=karar,
               veri=dict(uretim=ARAJ.get("uret"), esdegerlik=ARAJ.get("esdeger"), nedensellik=ARAJ.get("nedensellik"),
                         ozellikler=OZ, ozellik_aciklama=OZ_ACIKLAMA, kat_sinirlari=md["sinir"], evren_ozet=md["evren_ozet"]),
               madencilik=dict(tablo=tab, adaylar=md["adaylar"],
                               tek_degiskenli={k: {kk: vv for kk, vv in v.items()} for k, v in md["tek_degiskenli"].items()}),
               sonradan=ARAJ.get("sonradan"),
               valid=ARAJ.get("valid") or "CALISTIRILMADI: 0 aday (aday kurali (i)-(vi) hicbir konfigurasyonda saglanmadi); VALID verisine dokunulmadi")
    json.dump(_json(out), open(os.path.join(LAB, "sonuc_ict_ozellik.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# =============================================================== ana
def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return _json(o.tolist())
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", default="hepsi")
    ap.add_argument("--coin", type=int, default=98)
    ap.add_argument("--onek", default="tam")
    ap.add_argument("--isci", type=int, default=4)
    ap.add_argument("--seed", type=int, default=N_SEED)
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--valid", action="store_true", help="aday varsa VALID'i TEK SEFER calistir")
    a = ap.parse_args()
    syms = H.semboller_1h()[:a.coin]
    os.makedirs(ARA, exist_ok=True)
    ara_json = os.path.join(ARA, f"{a.onek}_ara.json")
    ARAJ = json.load(open(ara_json, encoding="utf-8")) if os.path.exists(ara_json) else {}

    def ara_yaz():
        json.dump(_json(ARAJ), open(ara_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if a.asama in ("esdeger",) or (a.asama == "hepsi" and "esdeger" not in ARAJ):
        e = esdeger(syms[:10])
        print("ESDEGERLIK:", e, flush=True)
        ARAJ["esdeger"] = e; ara_yaz()
        H.kaydet(AILE, "esdegerlik_testi", dict(coin=10, tfs=TFS), "gelistirme", e, "ict_sinyaller == bt.detect == ict2 C0_base")
    if a.asama in ("uret", "hepsi"):
        oz = uret(syms, a.onek, a.isci, a.seed)
        if oz:
            ARAJ.setdefault("uret", {}).update(oz); ara_yaz()
    if a.asama in ("nedensellik",) or (a.asama == "hepsi" and "nedensellik" not in ARAJ):
        nd = nedensellik(a.onek, syms)
        print("NEDENSELLIK:", nd, flush=True)
        ARAJ["nedensellik"] = nd; ara_yaz()
        H.kaydet(AILE, "nedensellik_testi", dict(n=nd["test"]), "gelistirme", nd, "veri sinyal barinda kesilince ayni ozellik")
    if a.asama in ("madencilik", "hepsi", "valid"):
        t0 = time.time()
        R = madencilik(a.onek, a.isci, a.seed, a.perm)
        tablo, adaylar = degerlendir(R)
        bet = betimleyici_tablo(R, a.seed)
        print(f"madencilik {time.time()-t0:.0f}s ; adaylar: {adaylar}", flush=True)
        for ad in KONFIGLER:
            x = tablo[ad]
            print(f"  {ad:16s} ICT n={x['ict']['n']:5d} wins={x['ict']['wins']} kazanc={x['ict']['kazanc']} katpoz={x['ict']['kat_pozitif']} "
                  f"P1%={x['P1_kazanc_p5_50_95']} P2%={x['P2_kazanc_p5_50_95']} perm_p={x['perm_p']} gecti={x['gecti']}", flush=True)
            H.kaydet(AILE, f"TRAIN_OOF|{ad}", dict(konfig=ad, kat=4, bosluk_gun=7, seed=a.seed, perm=a.perm, coin=a.coin),
                     "train", _json(x), "kat-disi filtre; plasebo P1/P2 + permutasyon karsilastirmasi")
        for oz_ad, v in R["uni_tab"].items():
            H.kaydet(AILE, f"TRAIN_tek|{oz_ad}", dict(ozellik=oz_ad), "train", _json({k: v.get(k) for k in v if k != "dilimler"}),
                     "tek degiskenli 5li dilim")
        H.kaydet(AILE, "betimleyici", dict(), "train", _json(bet), "kazanan vs kaybeden, ICT vs P1/P2")
        H.kaydet(AILE, "evren_ozet", dict(), "train", _json(R["evren_ozet"]), "ICT / P1 / P2 TRAIN ozet")
        ARAJ["madencilik"] = dict(sinir=[datetime.fromtimestamp(x / 1e3, timezone.utc).strftime("%Y-%m-%d") for x in R["sinir"]],
                                  ict_n=R["ict_n"], ict_base=R["ict"]["_base"], tablo=tablo, adaylar=adaylar,
                                  tek_degiskenli=R["uni_tab"], betimleyici=bet, evren_ozet=R["evren_ozet"],
                                  plasebo_base={u: [R["PL"][u][s]["_base"] for s in R["PL"][u]] for u in (1, 2)})
        ara_yaz()
        if a.valid and adaylar:
            if "valid" in ARAJ:
                print("VALID ZATEN CALISTIRILDI - tekrar yok", flush=True)
            else:
                sira = sorted(adaylar, key=lambda k: -tablo[k]["ict"]["kazanc"])[:3]
                ARAJ["valid"] = valid_calistir(a.onek, syms, sira, R, a.isci, a.seed); ara_yaz()
                print("VALID:", json.dumps(_json(ARAJ["valid"]))[:3000], flush=True)
    if a.asama == "sonradan":
        for oz_ad, dl in (("fvg_atr", 4), ("rsi14_hiz", 0), ("hacim_bos", 2)):
            r = sonradan(a.onek, oz_ad, dl, a.seed, a.perm)
            print("SONRADAN", json.dumps(_json(r)), flush=True)
            ARAJ.setdefault("sonradan", {})[f"{oz_ad}|{dl}"] = r; ara_yaz()
            H.kaydet(AILE, f"SONRADAN|{oz_ad}|dilim{dl}", dict(ozellik=oz_ad, dilim=dl), "train", _json(r),
                     "post-hoc sabit kural, plasebo+permutasyon; aday degil")
    print("bitti", flush=True)
