"""ict2 - ICT'nin YAPISAL varyantlari + bilgi-icerigi (kontrol) deneyi.

Aile: ict2. Dilimler: 4h ve 1h (harness.veri_1h + resample, kaynak="1h").
Parametre taramasi YOK (P0 = bt.py'deki canli-esdeger parametreler, sabit). Degisen: YAPI.

Konfigurasyonlar
  C0  base       : bt.detect esdegeri (BOS+FVG+OB, limit OB50, TP 5R)  -- referans
  C1  kontrol    : RASTGELE bar + RASTGELE yon, giris = son barin %50'si, stop = bar dibi,
                   ayni limit/stop/TP/pending/cooldown mekanigi. Bilgi-icerigi testi.
  C2  htf_btc    : C0 + sadece BTC gunluk MA50 rejimi yonunde
  C3  htf_coin   : C0 + sadece coinin kendi gunluk MA50 yonunde
  C4  htf_ikisi  : C0 + BTC VE coin gunluk MA50 ayni yonde
  C5  fvg50_5R   : giris FVG'nin %50'si (OB yerine), stop FVG disi, TP 5R
  C6  fvg50_swng : giris FVG %50, stop FVG disi, TP = ustteki/alttaki ilk likidite (swing)
  C7  sweep_2R   : likidite suprusu (onceki swing altina fitil + ustunde kapanis) piyasa girisi, TP 2R
  C8  sweep_3R   : ayni, TP 3R
  C9  4h_btc_mkt : sadece 4h, sadece BTC rejim yonunde, limit yerine PIYASA girisi (TP 5R)

Kullanim:  PYTHONIOENCODING=utf-8 python ict2.py --coin 98 --bolum train
"""
import os, sys, json, argparse, hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
ARACLAR = os.path.dirname(LAB)
sys.path.insert(0, LAB); sys.path.insert(0, ARACLAR)
import harness as H
import bt

P0 = dict(bt.P0)          # DEGISTIRILMEZ
TFS = ["4h", "1h"]
KONFIG = ["C0_base", "C1_kontrol", "C2_htf_btc", "C3_htf_coin", "C4_htf_ikisi",
          "C5_fvg50_5R", "C6_fvg50_swing", "C7_sweep_2R", "C8_sweep_3R", "C9_4h_btc_mkt"]
KONTROL_SEED = 20


# ------------------------------------------------------------------ yardimcilar
def _bars(sym, tf):
    a = H.veri_1h(sym)
    if len(a) < 1000:
        return None
    b = H.resample(a, tf)
    return b if len(b) >= 300 else None


_REJ = {}


def _coin_rejim(sym, N=50):
    """(gun_ot, +1/-1) coinin kendi gunluk MA50 rejimi; o gunun KAPANISINDA bilinir"""
    if sym in _REJ:
        return _REJ[sym]
    try:
        a = H.veri_1d(sym)
    except Exception:
        a = None
    if a is None or len(a) < N + 5:
        a = H.resample(H.veri_1h(sym), "1d")
    r = None
    if len(a) >= N + 5:
        c = a[:, 4]
        m = np.convolve(c, np.ones(N) / N, "full")[:len(c)]
        m[:N - 1] = np.nan
        r = (a[:, 0], np.where(c > m, 1, -1))
    _REJ[sym] = r
    return r


def _rej_at(rej, t_ms, N=50):
    if rej is None:
        return 0
    ots, v = rej
    k = np.searchsorted(ots, t_ms - 86400_000, side="right") - 1
    return int(v[k]) if k >= N else 0


def _yapi(d, i, side, P):
    """bt.detect'in YAPISAL kismi (swing/BOS/displacement/bayatlik/FVG/OB). Kabul kumesi bt ile ayni."""
    o, h, l, c, atr = d["o"], d["h"], d["l"], d["c"], d["atr"]
    if i < 80 or np.isnan(atr[i]):
        return None
    idx = d["sh_idx"] if side == 1 else d["sl_idx"]
    px = h if side == 1 else l
    lim = i - P["SWING_K"]
    m = np.searchsorted(idx, lim)
    if m < 2:
        return None
    son_swing = int(idx[m - 1]); swing_seviye = px[son_swing]
    bos_i = None
    for j in range(max(0, i - P["LOOKBACK_BOS"]), i):
        if j <= son_swing:
            continue
        if (c[j] > swing_seviye) if side == 1 else (c[j] < swing_seviye):
            bos_i = j; break
    if bos_i is None:
        return None
    if abs(c[bos_i] - o[bos_i]) < P["DISP_MULT"] * atr[bos_i]:
        return None
    if (i - bos_i) > P["MAX_BOS_YAS"]:
        return None
    fvg = None
    for j in range(max(bos_i - 2, 2), min(bos_i + 4, i + 1)):
        if side == 1 and l[j] > h[j - 2]:
            fvg = (h[j - 2], l[j]); break
        if side == -1 and h[j] < l[j - 2]:
            fvg = (h[j], l[j - 2]); break
    if fvg is None:
        return None
    ob = None
    for j in range(bos_i, max(bos_i - P["OB_ARAMA"], 0), -1):
        if side == 1 and c[j] < o[j]:
            ob = (l[j], h[j]); break
        if side == -1 and c[j] > o[j]:
            ob = (l[j], h[j]); break
    if ob is None:
        return None
    return dict(bos_i=bos_i, fvg=fvg, ob=ob, swing=float(swing_seviye), son_swing=son_swing, m=m)


def _bolge_plan(d, i, side, lo, hi, P):
    """OB/FVG bolgesi -> limit plan (giris = bolgenin %50'si, stop = bolge disi)"""
    entry = (lo + hi) / 2.0
    if side == 1:
        stop = lo * 0.999
        if stop >= entry or (entry - stop) / entry < P["MIN_STOP_PCT"]:
            stop = entry * (1 - P["MIN_STOP_PCT"])
        risk = entry - stop
    else:
        stop = hi * 1.001
        if stop <= entry or (stop - entry) / entry < P["MIN_STOP_PCT"]:
            stop = entry * (1 + P["MIN_STOP_PCT"])
        risk = stop - entry
    if risk <= 0:
        return None
    sp = risk / entry
    if sp < P["MIN_STOP_PCT"] or sp > P["MAX_STOP_PCT"]:
        return None
    son = d["c"][i]
    if abs(entry - son) / risk > P["MAX_ENTRY_DIST_R"]:
        return None
    if side == 1 and (son < entry or son < stop):
        return None
    if side == -1 and (son > entry or son > stop):
        return None
    return entry, stop, risk


def _swing_hedef(d, i, side, entry, risk, P):
    """entry'nin uzerinde/altinda henuz alinmamis ILK likidite (teyitli swing seviyesi)"""
    idx = d["sh_idx"] if side == 1 else d["sl_idx"]
    px = d["h"] if side == 1 else d["l"]
    lim = i - P["SWING_K"]
    m = np.searchsorted(idx, lim)
    if m < 1:
        return None
    lv = px[idx[:m]]
    if side == 1:
        cand = lv[lv > entry + 0.5 * risk]
        return float(cand.min()) if len(cand) else None
    cand = lv[lv < entry - 0.5 * risk]
    return float(cand.max()) if len(cand) else None


# ------------------------------------------------------------------ sinyal ureticileri
def _sinyaller(sym, tf, bars, d, P):
    """tek gecis: tum yapisal varyantlarin sinyallerini uretir"""
    out = {k: [] for k in KONFIG}
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    n = len(c)
    tfms = H.TF_H[tf] * 3600_000
    rej_coin = _coin_rejim(sym)
    for i in range(80, n - 1):
        sig_close = float(bars[i, 0]) + tfms
        rb = rc = None
        for side in (1, -1):
            y = _yapi(d, i, side, P)
            if y is None:
                continue
            bos_i, ob, fvg = y["bos_i"], y["ob"], y["fvg"]
            # ---- OB tabanli (C0/C2/C3/C4/C9) : bolge dolmus mu?
            ob_ok = True
            if i > bos_i + 1:
                if side == 1 and l[bos_i + 1:i].min() < ob[0]:
                    ob_ok = False
                if side == -1 and h[bos_i + 1:i].max() > ob[1]:
                    ob_ok = False
            if ob_ok:
                pl = _bolge_plan(d, i, side, ob[0], ob[1], P)
                if pl:
                    e, st, risk = pl
                    base = dict(i=i, side=side, entry=e, stop=st, tp=e + side * P["TP_R"] * risk,
                                pending_h=P["PENDING_H"], max_days=P["MAX_DAYS"])
                    out["C0_base"].append(base)
                    if rb is None:
                        rb = H.rejim_at(sig_close); rc = _rej_at(rej_coin, sig_close)
                    if rb == side:
                        out["C2_htf_btc"].append(dict(base))
                    if rc == side:
                        out["C3_htf_coin"].append(dict(base))
                    if rb == side and rc == side:
                        out["C4_htf_ikisi"].append(dict(base))
                    # C9: piyasa girisi (limit yok), stop OB disi, TP 5R -- risk son fiyattan
                    if tf == "4h" and rb == side:
                        rk = abs(c[i] - st)
                        sp = rk / c[i]
                        if P["MIN_STOP_PCT"] <= sp <= P["MAX_STOP_PCT"]:
                            out["C9_4h_btc_mkt"].append(dict(i=i, side=side, entry=None, stop=st,
                                                             tp=c[i] + side * P["TP_R"] * rk,
                                                             max_days=P["MAX_DAYS"]))
            # ---- FVG tabanli (C5/C6)
            flo, fhi = fvg
            fvg_ok = True
            if i > bos_i + 1:
                if side == 1 and l[bos_i + 1:i].min() < flo:
                    fvg_ok = False
                if side == -1 and h[bos_i + 1:i].max() > fhi:
                    fvg_ok = False
            if fvg_ok:
                pl = _bolge_plan(d, i, side, flo, fhi, P)
                if pl:
                    e, st, risk = pl
                    out["C5_fvg50_5R"].append(dict(i=i, side=side, entry=e, stop=st,
                                                   tp=e + side * P["TP_R"] * risk,
                                                   pending_h=P["PENDING_H"], max_days=P["MAX_DAYS"]))
                    hed = _swing_hedef(d, i, side, e, risk, P)
                    if hed is not None:
                        out["C6_fvg50_swing"].append(dict(i=i, side=side, entry=e, stop=st, tp=hed,
                                                          pending_h=P["PENDING_H"], max_days=P["MAX_DAYS"]))
        # ---- C7/C8 likidite suprusu (yapidan bagimsiz)
        sw = _sweep(d, i, P)
        if sw:
            side, st, rk = sw
            for ad, R in (("C7_sweep_2R", 2.0), ("C8_sweep_3R", 3.0)):
                out[ad].append(dict(i=i, side=side, entry=None, stop=st,
                                    tp=c[i] + side * R * rk, max_days=P["MAX_DAYS"]))
    return out


def _sweep(d, i, P):
    """onceki teyitli swing low ALTINA fitil + USTUNDE kapanis -> LONG (simetrik SHORT)"""
    l, h, c = d["l"], d["h"], d["c"]
    if i < 80:
        return None
    k = P["SWING_K"]; lim = i - k
    for side in (1, -1):
        idx = d["sl_idx"] if side == 1 else d["sh_idx"]
        px = l if side == 1 else h
        m = np.searchsorted(idx, lim)
        if m < 1:
            continue
        lvl = px[idx[m - 1]]
        if side == 1 and l[i] < lvl and c[i] > lvl:
            st = l[i] * 0.999
        elif side == -1 and h[i] > lvl and c[i] < lvl:
            st = h[i] * 1.001
        else:
            continue
        rk = abs(c[i] - st)
        if rk <= 0:
            continue
        sp = rk / c[i]
        if sp < P["MIN_STOP_PCT"] or sp > P["MAX_STOP_PCT"]:
            continue
        return side, st, rk
    return None


def _kontrol_sinyaller(d, bars, tf, n_sig, seed, P, mod="rastgele", i_list=None):
    """Kontrol girisleri: giris = o barin %50'si, stop = bar dibi/tepesi (OB yerine),
    ayni limit/stop/TP/pending/cooldown mekanigi, ayni geometrik gecerlilik filtreleri.
      mod='rastgele' : rastgele bar + rastgele yon
      mod='htf'      : rastgele bar, yon = o andaki BTC gunluk MA50 rejimi
      mod='zaman'    : ICT'nin TAM AYNI barlari, rastgele yon (zaman secimini sabitler)"""
    l, h, c = d["l"], d["h"], d["c"]
    n = len(c)
    tfms = H.TF_H[tf] * 3600_000
    rng = np.random.default_rng(seed)
    sig = []
    if mod == "zaman":
        havuz = list(i_list or [])
        rng.shuffle(havuz)
        aday = havuz
    else:
        aday = [int(x) for x in rng.integers(80, n - 1, size=n_sig * 60 + 500)]
    for i in aday:
        if len(sig) >= n_sig:
            break
        if mod == "htf":
            side = H.rejim_at(float(bars[i, 0]) + tfms)
            if side == 0:
                continue
        else:
            side = 1 if rng.random() < 0.5 else -1
        pl = _bolge_plan(d, i, side, l[i], h[i], P)
        if pl is None:
            continue
        e, st, risk = pl
        sig.append(dict(i=i, side=side, entry=e, stop=st, tp=e + side * P["TP_R"] * risk,
                        pending_h=P["PENDING_H"], max_days=P["MAX_DAYS"]))
    sig.sort(key=lambda s: s["i"])
    return sig


# ------------------------------------------------------------------ isci
def _isci(arg):
    sym, tf, kontrol = arg
    try:
        bars = _bars(sym, tf)
        if bars is None:
            return {}
        d = bt.hazirla(bars, P0)
        S = _sinyaller(sym, tf, bars, d, P0)
        res = {}
        for ad, sg in S.items():
            if not sg:
                continue
            sg.sort(key=lambda s: s["i"])
            res[ad] = H.sim_islem(bars, sg, tf=tf, cooldown_h=P0["COOLDOWN_H"])
        if kontrol:
            n_sig = len(S["C0_base"])
            i_list = [s["i"] for s in S["C0_base"]]
            if n_sig >= 3:
                for mod, ad in (("rastgele", "C1_kontrol"), ("htf", "C1b_kontrol_htf"), ("zaman", "C1c_kontrol_zaman")):
                    for sd in range(KONTROL_SEED):
                        seed = int(hashlib.md5(f"{sym}|{tf}|{mod}|{sd}".encode()).hexdigest()[:8], 16)
                        ks = _kontrol_sinyaller(d, bars, tf, n_sig, seed, P0, mod, i_list)
                        if ks:
                            res[f"{ad}#{sd}"] = H.sim_islem(bars, ks, tf=tf, cooldown_h=P0["COOLDOWN_H"])
        return res
    except Exception as ex:
        return {"__hata__": [f"{sym} {tf}: {ex}"]}


# ------------------------------------------------------------------ istatistik
def _boot_fark(a, b, n=4000):
    """mean(a) - mean(b) bootstrap dagilimi -> (fark, GA95, GA99, p_iki_yonlu)"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 20 or len(b) < 20:
        return None
    ra = np.random.default_rng(7)
    d = np.empty(n)
    for k in range(n):
        d[k] = a[ra.integers(0, len(a), len(a))].mean() - b[ra.integers(0, len(b), len(b))].mean()
    d.sort()
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(fark=round(float(a.mean() - b.mean()), 4),
                ga95=[round(float(d[int(n * .025)]), 4), round(float(d[int(n * .975)]), 4)],
                ga99=[round(float(d[int(n * .005)]), 4), round(float(d[int(n * .995)]), 4)],
                p=round(float(p), 4))


def _net(trades):
    return [t["net"] for t in trades if t["res"] != "cancelled"]


# ------------------------------------------------------------------ ana
def calis(coin=98, tfs=None, kontrol=True, isci=8):
    tfs = tfs or TFS
    syms = H.semboller_1h()[:coin]
    T = {}          # (tf, ad) -> trades
    hatalar = []
    for tf in tfs:
        args = [(s, tf, kontrol) for s in syms]
        with ProcessPoolExecutor(max_workers=isci) as ex:
            for r in ex.map(_isci, args, chunksize=2):
                for ad, tr in r.items():
                    if ad == "__hata__":
                        hatalar += tr; continue
                    T.setdefault((tf, ad), []).extend(tr)
        print(f"[{tf}] bitti. konfig: {sorted({a for (t, a) in T if t == tf and '#' not in a})}", flush=True)
    if hatalar:
        print("HATA:", hatalar[:5], f"(toplam {len(hatalar)})", flush=True)
    return T


def ozet(T, bolum="train"):
    sat = []
    for (tf, ad) in sorted(T):
        if "#" in ad:
            continue
        r = H.rapor(T[(tf, ad)], bolum, "1h")
        sat.append((tf, ad, r))
    return sat


def kontrol_testi(T, tf, bolum, ict_ad="C0_base"):
    """ICT net-R dagilimi ile kontrol (rastgele giris) dagilimini karsilastir"""
    ict = _net(H.bolum_filtre(T.get((tf, ict_ad), []), bolum, "1h"))
    cik = dict(tf=tf, bolum=bolum, ict=ict_ad, ict_n=len(ict),
               ict_ort=round(float(np.mean(ict)), 4) if ict else None)
    for ad in ("C1_kontrol", "C1b_kontrol_htf", "C1c_kontrol_zaman"):
        seeds, pool = [], []
        for (t, a) in T:
            if t == tf and a.startswith(ad + "#"):
                x = _net(H.bolum_filtre(T[(t, a)], bolum, "1h"))
                if x:
                    seeds.append(float(np.mean(x))); pool += x
        if not pool:
            continue
        s = np.array(seeds)
        cik[ad] = dict(havuz_n=len(pool), ort=round(float(np.mean(pool)), 4),
                       seed_ort=[round(float(np.percentile(s, q)), 4) for q in (5, 50, 95)],
                       fark_ict_eksi_kontrol=_boot_fark(ict, pool))
    return cik


def ana(coin=98, tfs=None, isci=10, valid_adaylar=None):
    tfs = tfs or TFS
    T = calis(coin, tfs, True, isci)
    import pickle
    pickle.dump({f"{k[0]}|{k[1]}": v for k, v in T.items()}, open(os.path.join(LAB, "_ict2_cache.pkl"), "wb"))

    # ---- 1) her konfigurasyonu deftere yaz (train)
    tablo = {}
    for (tf, ad) in sorted(T):
        if "#" in ad:
            continue
        r = H.rapor(T[(tf, ad)], "train", "1h")
        tablo[f"{tf}|{ad}"] = r
        H.kaydet("ict2", f"{tf}|{ad}", dict(dilim=tf, konfig=ad, P="bt.P0 sabit"), "train", r,
                 "yapisal varyant; parametre taramasi yok")
    # kontrol havuzlari da deftere
    kt = {}
    for tf in tfs:
        for bol in ("train", "gelistirme"):
            k = kontrol_testi(T, tf, bol)
            kt[f"{tf}|{bol}"] = k
            H.kaydet("ict2", f"{tf}|KONTROL", dict(dilim=tf, konfig="kontrol_karsilastirma"), bol, k,
                     "rastgele giris kontrolu (bilgi-icerigi testi)")

    # ---- 2) VALID: en fazla 3 aday, TEK sefer
    valid = {}
    for key in (valid_adaylar or []):
        tf, ad = key.split("|")
        r = H.rapor(T[(tf, ad)], "valid", "1h")
        valid[key] = r
        H.kaydet("ict2", key, dict(dilim=tf, konfig=ad), "valid", r, "TEK SEFERLIK valid")
    return T, tablo, kt, valid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin", type=int, default=98)
    ap.add_argument("--tf", default="4h,1h")
    ap.add_argument("--isci", type=int, default=10)
    ap.add_argument("--valid", default="")     # "1h|C2_htf_btc,1h|C4_htf_ikisi"
    a = ap.parse_args()
    T, tablo, kt, valid = ana(a.coin, a.tf.split(","), a.isci,
                              [x for x in a.valid.split(",") if x])
    print("\n=== TRAIN ===")
    for k, r in tablo.items():
        print(k, json.dumps(r), flush=True)
    print("\n=== KONTROL ===")
    for k, v in kt.items():
        print(k, json.dumps(v), flush=True)
    print("\n=== VALID ===")
    for k, r in valid.items():
        print(k, json.dumps(r), flush=True)
