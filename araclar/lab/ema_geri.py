"""AILE: ema_geri - Yukselis sonrasi EMA(hizli)-EMA(yavas) bandina geri cekilip tekrar yukselen LONG islemler.

Kaynak / bolum:
  4h : harness.veri_1h + resample('4h'), tum 1h evreni, kaynak="1h" -> TRAIN < 2025-07, VALID 2025-07..12
  1d : harness.veri_1d, likit evren (sinyal aninda 30g ort USDT hacmi (kolon 5) >= 5M), kaynak="1d" -> TRAIN < 2025, VALID 2025
  HOLDOUT (2026) KAPALI: LAB_HOLDOUT asla ayarlanmaz; veri yalniz harness fonksiyonlarindan.

KURALLAR (sinyal i barinin KAPANISINDA, giris i+1 acilisi, sim_islem entry=None):
 1 TREND  : EMAf[i] > EMAs[i]; EMAs[i] > EMAs[i-10]; max(close[i-R..i]) >= close[i-R]*(1+Y)
 2 GERI   : pencere W = [i-P, i-1]. W icinde zirveden (argmax high[i-R..i-1]) SONRA en az bir k:
            low[k] <= EMAf[k] ve close[k] >= EMAs[k]-0.5*ATR14[k]; W'de hicbir close < EMAs-0.5*ATR;
            i - zirve >= 3
 3 TETIK  : close[i] > EMAf[i] ve close[i] > high[i-1]
 4 STOP   : min(low[W]) - 0.5*ATR14[i]; stop % < 0.5 veya > 15 -> atla
 5 CIKIS  : tp2 | tp3 | ema50 (ilk close < EMAs) | ema20 (ilk close < EMAf); max_days 4h=10, 1d=60
 6 FILTRE : f0 yok | f1 BTC gunluk MA50 rejimi (harness mantigi, son kapanmis gun) | f2 f1 + coin gunluk close > MA50
PLASEBO (eslesmis): ayni coin, ayni bolum; TREND + ayni filtre (+1d likidite) saglanan barlardan strateji sinyal
 sayisi kadar rastgele bar; stop = close*(1 - coinin o bolumdeki strateji medyan stop %'si); ayni cikis. 20 seed.
 Tohum kurgusu kirilim_gunluk cfg['plasebo'] ile ayni. Winsorize/top%1: rejim.saglam().

Kullanim:
  PYTHONIOENCODING=utf-8 python ema_geri.py --asama duman --coin 10
  ... --asama tarama              # f0 + tp3, EMA x Y x P, 4h ve 1d (TRAIN)
  ... --asama bolge               # en iyi bolgede cikis x filtre (TRAIN)
  ... --asama degerlendir         # aday kurali, _ema_geri_train.json uzerinden
  ... --asama valid --adaylar a,b # TEK SEFER
"""
import os, sys, json, argparse, itertools, time
import numpy as np
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from rejim import saglam          # 3R-winsorize (clip -1.5..3), top %1 payi

AILE = "ema_geri"
LAB = os.path.dirname(os.path.abspath(__file__))
HAM = os.path.join(LAB, "_ema_geri_ham.json")
SONUC = os.path.join(LAB, "sonuc_ema_geri.json")
MIN_VOL = 5e6
R_LOOK = 20
N_SEED = 20
EMA_CIFT = [(15, 40), (20, 50), (25, 60)]
Y_GRID = {"4h": [0.10, 0.15, 0.20], "1d": [0.15, 0.25, 0.35]}
P_GRID = [7, 10, 15]
MAX_DAYS = {"4h": 10, "1d": 60}
KAYNAK = {"4h": "1h", "1d": "1d"}


# ---------------------------------------------------------------- gostergeler (nedensel)
def ema(x, n):
    a = 2.0 / (n + 1); out = np.empty(len(x)); m = x[0]
    for i, v in enumerate(x):
        m = a * v + (1 - a) * m; out[i] = m
    return out


def roll_mean(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.concatenate([[0.0], np.cumsum(x)]); out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def atr14(h, l, c):
    pc = np.concatenate([[c[0]], c[:-1]])
    return roll_mean(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), 14)


def btc_rejim_vec(T):
    """harness.rejim_at(T, 50) ile birebir, vektorlestirilmis (T = sinyal kapanis ms)."""
    ots, r, _ = H.btc_ma_rejim(50)
    k = np.searchsorted(ots, T - 86400_000, side="right") - 1
    kk = np.clip(k, 0, len(r) - 1)
    return np.where(k >= 50, r[kk], 0)


def ad_of(c):
    return f"{c['tf']}_e{c['ema'][0]}-{c['ema'][1]}_Y{c['Y']}_P{c['P']}_{c['cikis']}_{c['filtre']}"


# ---------------------------------------------------------------- coin hazirlik
def yukle(tf, sym):
    if tf == "4h":
        a1 = H.veri_1h(sym)
        a = H.resample(a1, "4h"); d = H.resample(a1, "1d")
    else:
        a = H.veri_1d(sym); d = a
    if len(a) < 150 or not np.isfinite(a[:, 1:5]).all():
        return None
    ot, h, l, c = a[:, 0], a[:, 2], a[:, 3], a[:, 4]
    tfms = H.TF_H[tf] * 3600_000
    T = ot + tfms
    # coin gunluk MA50 (son kapanmis gun)
    dc = d[:, 4]; dma = roll_mean(dc, 50); dust = dc > np.nan_to_num(dma, nan=np.inf)
    k = np.searchsorted(d[:, 0], T - 86400_000, side="right") - 1
    coin_ma = np.where(k >= 49, dust[np.clip(k, 0, len(d) - 1)], False)
    liq = np.ones(len(c), bool)
    if tf == "1d":
        v30 = roll_mean(a[:, 5], 30); liq = np.isfinite(v30) & (v30 >= MIN_VOL)
    return dict(a=a, ot=ot, T=T, h=h, l=l, c=c, atr=atr14(h, l, c), btc=btc_rejim_vec(T) == 1,
                coin_ma=coin_ma, liq=liq, ema={})


def ilk_sonra(cond):
    """nxt[j] = j'den itibaren (j dahil) cond'un ilk dogru oldugu indeks, yoksa -1"""
    n = len(cond); nxt = np.full(n + 1, -1, np.int64)
    for j in range(n - 1, -1, -1):
        nxt[j] = j if cond[j] else nxt[j + 1]
    return nxt


def ema_hazir(D, ef, es):
    key = (ef, es)
    if key not in D["ema"]:
        c = D["c"]; e1 = ema(c, ef); e2 = ema(c, es)
        D["ema"][key] = dict(e1=e1, e2=e2, nx1=ilk_sonra(c < e1), nx2=ilk_sonra(c < e2))
    return D["ema"][key]


def trend_maske(D, E, Y, R=R_LOOK):
    c = D["c"]; n = len(c); e1, e2 = E["e1"], E["e2"]
    m = np.zeros(n, bool)
    i = np.arange(n)
    ok = i >= max(R, 10)
    e2l = np.full(n, np.inf); e2l[10:] = e2[:-10]
    cl = np.full(n, np.inf); cl[R:] = c[:-R]
    mx = np.full(n, -np.inf)
    sw = np.lib.stride_tricks.sliding_window_view(c, R + 1).max(axis=1)   # sw[j] = max c[j..j+R]
    mx[R:] = sw
    m = ok & (e1 > e2) & (e2 > e2l) & (mx >= cl * (1 + Y))
    return m


def filtre_maske(D, f):
    if f == "f0":
        return D["liq"].copy()
    if f == "f1":
        return D["liq"] & D["btc"]
    return D["liq"] & D["btc"] & D["coin_ma"]


def cikis_bar(E, i, cikis):
    if cikis == "ema50":
        j = E["nx2"][i + 1]; return int(j) if j >= 0 else None
    if cikis == "ema20":
        j = E["nx1"][i + 1]; return int(j) if j >= 0 else None
    return None


def sinyal_yap(i, c_i, stop, cikis, E, tf):
    risk = c_i - stop
    tp = None
    if cikis == "tp2":
        tp = float(c_i + 2 * risk)
    elif cikis == "tp3":
        tp = float(c_i + 3 * risk)
    return dict(i=int(i), side=1, entry=None, stop=float(stop), tp=tp,
                cikis_i=cikis_bar(E, i, cikis), max_days=MAX_DAYS[tf])


def strateji_sinyal(D, cfg, bmask):
    tf = cfg["tf"]; ef, es = cfg["ema"]; Y, P = cfg["Y"], cfg["P"]
    E = ema_hazir(D, ef, es)
    c, h, l, a = D["c"], D["h"], D["l"], D["atr"]; e1, e2 = E["e1"], E["e2"]
    n = len(c)
    warm = max(2 * es, R_LOOK + P + 15, 60)
    tr = trend_maske(D, E, Y) & filtre_maske(D, cfg["filtre"])
    tr[:warm] = False; tr[n - 2:] = False
    trig = np.zeros(n, bool)
    trig[1:] = (c[1:] > e1[1:]) & (c[1:] > h[:-1])
    aday = np.flatnonzero(tr & trig & bmask & np.isfinite(a) & (a > 0))
    lim = e2 - 0.5 * a
    touch = (l <= e1) & (c >= lim)
    bozuk = c < lim
    sig = []; sps = []
    for i in aday:
        i = int(i)
        w0 = i - P
        if bozuk[w0:i].any():
            continue
        pk = (i - R_LOOK) + int(np.argmax(h[i - R_LOOK:i]))
        if i - pk < 3:
            continue
        k0 = max(w0, pk + 1)
        if not touch[k0:i].any():
            continue
        stop = float(l[w0:i].min() - 0.5 * a[i])
        sp = (c[i] - stop) / c[i]
        if sp < 0.005 or sp > 0.15:
            continue
        sig.append(sinyal_yap(i, c[i], stop, cfg["cikis"], E, tf)); sps.append(sp)
    return sig, sps, tr


def plasebo_sinyal(D, cfg, tr, bmask, kac, med_sp, seed):
    E = ema_hazir(D, *cfg["ema"])
    ok = np.flatnonzero(tr & bmask)
    if kac == 0 or len(ok) == 0:
        return []
    rng = np.random.default_rng(int(seed) * 1000003 + (int(D["ot"][0]) % 100000))
    sec = np.sort(rng.choice(ok, size=min(kac, len(ok)), replace=False))
    c = D["c"]
    return [sinyal_yap(i, c[i], c[i] * (1 - med_sp), cfg["cikis"], E, cfg["tf"]) for i in sec]


RES = ["stopped", "target_done", "timeout", "cikis", "cancelled"]


def _kisa(tl, sym=None):
    """bellek: islem listesi -> kompakt diziler (t, net, r, res_kodu)"""
    return (np.array([t["t"] for t in tl], np.int64), np.array([t["net"] for t in tl], np.float64),
            np.array([t["r"] for t in tl], np.float64), np.array([RES.index(t["res"]) for t in tl], np.int8))


_BOS = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int8))


def _birlestir(parcalar):
    parcalar = [x for x in parcalar if len(x[0])]
    if not parcalar:
        return _BOS
    return tuple(np.concatenate([x[k] for x in parcalar]) for k in range(4))


def _dicts(A):
    o = np.argsort(A[0], kind="stable")
    return [dict(t=int(A[0][i]), res=RES[int(A[3][i])], net=float(A[1][i]), r=float(A[2][i])) for i in o]


def _isle(arg):
    tf, sym, cfgs, bolum = arg
    try:
        D = yukle(tf, sym)
    except Exception:
        return sym, {}
    if D is None:
        return sym, {}
    a0, b0 = (H.BOLUM_1H if KAYNAK[tf] == "1h" else H.BOLUM)[bolum]
    bmask = (D["T"] >= a0) & (D["T"] < b0)
    out = {}
    for cfg in cfgs:
        sig, sps, tr = strateji_sinyal(D, cfg, bmask)
        st = _kisa(H.sim_islem(D["a"], sig, tf=tf)) if sig else _BOS
        pl = []
        if sig:
            med = float(np.median(sps))
            for s in range(N_SEED):
                ps = plasebo_sinyal(D, cfg, tr, bmask, len(sig), med, s)
                pl.append(_kisa(H.sim_islem(D["a"], ps, tf=tf)) if ps else _BOS)
        else:
            pl = [_BOS for _ in range(N_SEED)]
        out[ad_of(cfg)] = dict(s=st, p=pl, nsig=len(sig), sp=sps)
    return sym, out


def kos(cfgs, bolum, coin=0, isci=8):
    """cfgs ayni tf'de olmali"""
    tf = cfgs[0]["tf"]
    syms = H.semboller_1h() if tf == "4h" else H.semboller_1d()
    if coin:
        syms = syms[:coin]
    res = {ad_of(c): dict(s=[], p=[[] for _ in range(N_SEED)], nsig=0, sp=[]) for c in cfgs}
    args = [(tf, s, cfgs, bolum) for s in syms]
    with ProcessPoolExecutor(max_workers=isci) as ex:
        for sym, d in ex.map(_isle, args, chunksize=4):
            for ad, x in d.items():
                R = res[ad]; R["s"].append(x["s"]); R["nsig"] += x["nsig"]; R["sp"] += x["sp"]
                for k in range(N_SEED):
                    R["p"][k].append(x["p"][k])
    for R in res.values():
        R["s"] = _birlestir(R["s"]); R["p"] = [_birlestir(q) for q in R["p"]]
    return res


# ---------------------------------------------------------------- istatistik
def _w3(tl):
    return np.clip(np.array([t["net"] for t in tl if t["res"] != "cancelled"], float), -1.5, 3.0)


def _ay(tl):
    return np.array([datetime.fromtimestamp(t["t"] / 1e3, timezone.utc).strftime("%Y-%m") for t in tl
                     if t["res"] != "cancelled"])


def fark_ga(st, pls, B=2000, rng=None):
    """strateji w3 ort - plasebo w3 ort. iid: strateji islemleri + rastgele bir seed'in islemleri yeniden ornekleme.
    ay_blok: takvim ayi kumeleri yeniden orneklenir (islem kumelenmesine karsi daha muhafazakar)."""
    rng = rng or np.random.default_rng(7)
    xs = _w3(st); P = [(_w3(p), _ay(p)) for p in pls if len(p) >= 5]
    if len(xs) < 5 or not P:
        return None, None, None
    iid = np.empty(B)
    for b in range(B):
        px = P[rng.integers(len(P))][0]
        iid[b] = xs[rng.integers(0, len(xs), len(xs))].mean() - px[rng.integers(0, len(px), len(px))].mean()
    # EslesMIS ay-blok: strateji ve plasebo AYNI takvim aylari ile yeniden orneklenir (ortak piyasa gurultusu iptal)
    ays = _ay(st); um = np.unique(ays)
    gi = {u: np.flatnonzero(ays == u) for u in um}
    Pg = [(px, {u: np.flatnonzero(pa == u) for u in um}) for px, pa in P]
    blk = np.empty(B)
    for b in range(B):
        sec = um[rng.integers(0, len(um), len(um))]
        px, pg = Pg[rng.integers(len(Pg))]
        idx = np.concatenate([gi[u] for u in sec])
        pidx = np.concatenate([pg[u] for u in sec])
        blk[b] = xs[idx].mean() - (px[pidx].mean() if len(pidx) else np.nan)
    blk = blk[np.isfinite(blk)]
    # ikincil (daha liberal): plasebo ortalamasi 20 seed HAVUZUNDAN tahmin edilir (tek seed gurultusu cift sayilmaz)
    hv = np.concatenate([px for px, _ in P]); hvd = np.empty(B)
    for b in range(B):
        hvd[b] = xs[rng.integers(0, len(xs), len(xs))].mean() - hv[rng.integers(0, len(hv), len(hv))].mean()
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
    return q(iid), q(blk), q(hvd)


def _yil_w3(tl):
    d = {}
    for t in tl:
        if t["res"] == "cancelled":
            continue
        y = datetime.fromtimestamp(t["t"] / 1e3, timezone.utc).year
        d.setdefault(y, []).append(min(3.0, max(-1.5, t["net"])))
    return {int(y): round(float(np.mean(v)), 3) for y, v in sorted(d.items())}


def ozetle(cfg, R, bolum):
    kay = KAYNAK[cfg["tf"]]
    R = dict(R, s=_dicts(R["s"]), p=[_dicts(q) for q in R["p"]])
    st = R["s"]
    rp = H.rapor(st, None, kay)       # zaten bolume filtreli
    sg = saglam(st)
    prp = [H.rapor(p, None, kay) for p in R["p"]]
    psg = [saglam(p) for p in R["p"]]
    okp = [i for i in range(N_SEED) if prp[i].get("ort_net") is not None and "wins3" in psg[i]]
    m = lambda key, L: round(float(np.mean([L[i][key] for i in okp])), 4) if okp else None
    pl = dict(n_ort=round(float(np.mean([prp[i]["n"] for i in okp])), 1) if okp else 0,
              ort_net=m("ort_net", prp), ort_brut=m("ort_brut", prp), win=m("win", prp),
              wins3=m("wins3", psg),
              wins3_seed_min_max=[round(min(psg[i]["wins3"] for i in okp), 4), round(max(psg[i]["wins3"] for i in okp), 4)] if okp else None,
              top1_pay=round(float(np.nanmean([psg[i]["top1_pay"] if psg[i].get("top1_pay") is not None else np.nan for i in okp])), 3) if okp else None,
              yillar_w3=_yil_w3([t for p in R["p"] for t in p]))
    ga_iid, ga_blk, ga_hv = fark_ga(st, R["p"])
    w3 = sg.get("wins3")
    out = dict(rp)
    out.update(sinyal_ham=R["nsig"], medyan_stop_pct=round(float(np.median(R["sp"])), 4) if R["sp"] else None,
               wins3=w3, medyan=sg.get("medyan"), top1_pay=sg.get("top1_pay"), maxR=sg.get("maxR"),
               yillar_w3=_yil_w3(st), plasebo=pl,
               fark_w3=round(w3 - pl["wins3"], 4) if (w3 is not None and pl["wins3"] is not None) else None,
               fark_ga95=ga_iid, fark_ga95_ayblok=ga_blk, fark_ga95_havuz=ga_hv, fark_net=round(rp["ort_net"] - pl["ort_net"], 4)
               if (rp.get("ort_net") is not None and pl["ort_net"] is not None) else None)
    return out


def satir(ad, o):
    if o.get("ort_net") is None or o.get("wins3") is None:
        return f"  {ad:40s} n={o.get('n')} (yetersiz)"
    p = o["plasebo"]
    return (f"  {ad:40s} n={o['n']:5d} net={o['ort_net']:+.3f} brut={o['ort_brut']:+.3f} w3={o['wins3']:+.3f} "
            f"| pl n={p['n_ort']:.0f} net={p['ort_net']:+.3f} w3={p['wins3']:+.3f} | fark={o['fark_w3']:+.3f} "
            f"ga={o['fark_ga95']} blk={o['fark_ga95_ayblok']} top1={o['top1_pay']} win={o['win']:.2f} "
            f"y1={o['yari1']:+.3f} y2={o['yari2']:+.3f}")


ESKI_HATALI_NOT = "asama1 tarama f0+tp3"     # iptal edilen ilk kosu: esleSMEMIS ay-blok GA (hatali), yerine TEKRAR gecer


def ham_defterden():
    """_ema_geri_ham.json yoksa defterden kur: duman testleri ve hatali ilk kosu haric, (ad, bolum) basina SON kayit."""
    d = {}
    for x in H.defter_oku():
        if x.get("aile") != AILE or x.get("not_", "").startswith("duman") or x.get("not_") == ESKI_HATALI_NOT:
            continue
        key = x["ad"] if x["bolum"] == "train" else x["ad"] + "@" + x["bolum"]
        d[key] = dict(params=x["params"], bolum=x["bolum"], rapor=x["rapor"])
    return d


def ham_oku():
    if os.path.exists(HAM):
        return json.load(open(HAM, encoding="utf-8"))
    return ham_defterden()


def ham_yaz(d):
    json.dump(d, open(HAM, "w", encoding="utf-8"), indent=1, default=str)


def calistir(cfgs, bolum, coin, isci, not_, kaydet=True, prefix="", ham=None):
    t0 = time.time(); sonuc = {}
    for tf in ("4h", "1d"):
        cc = [c for c in cfgs if c["tf"] == tf]
        if not cc:
            continue
        res = kos(cc, bolum, coin, isci)
        for c in cc:
            ad = prefix + ad_of(c); o = ozetle(c, res[ad_of(c)], bolum)
            if kaydet:
                H.kaydet(AILE, ad, c, bolum, o, not_)
            sonuc[ad] = dict(params=c, bolum=bolum, rapor=o)
            print(satir(ad, o), flush=True)
        print(f"  [{tf}] {len(cc)} konfig, {time.time() - t0:.0f}s", flush=True)
        if ham is not None:                      # tf basina ara kayit (cokmeye karsi)
            for k, x in sonuc.items():
                ham[k if bolum == "train" else k + "@" + bolum] = x
            ham_yaz(ham)
    return sonuc


# ---------------------------------------------------------------- asamalar
def tarama_cfgs(tf):
    return [dict(tf=tf, ema=list(e), Y=Y, P=P, R=R_LOOK, cikis="tp3", filtre="f0")
            for e in EMA_CIFT for Y in Y_GRID[tf] for P in P_GRID]


def komsular(cfg, havuz):
    """tek bir parametrede (EMA / Y / P) +-1 adim farkli, ayni tf/cikis/filtre"""
    tf = cfg["tf"]; out = []
    ei = [list(e) for e in EMA_CIFT].index(list(cfg["ema"])); yi = Y_GRID[tf].index(cfg["Y"]); pi = P_GRID.index(cfg["P"])
    for dim in range(3):
        for s in (-1, 1):
            idx = [ei, yi, pi]; idx[dim] += s
            L = [len(EMA_CIFT), len(Y_GRID[tf]), len(P_GRID)]
            if not 0 <= idx[dim] < L[dim]:
                continue
            k = dict(cfg, ema=list(EMA_CIFT[idx[0]]), Y=Y_GRID[tf][idx[1]], P=P_GRID[idx[2]])
            ad = ad_of(k)
            if ad in havuz:
                out.append(ad)
    return out


def en_iyi_bolge(tf, ham):
    """komsu-ortalamali fark_w3 en yuksek merkez (tek konfig degil bolge)"""
    havuz = {ad: v for ad, v in ham.items() if v["params"]["tf"] == tf and v["params"]["cikis"] == "tp3"
             and v["params"]["filtre"] == "f0" and v["bolum"] == "train"}
    best, bs = None, -9
    for ad, v in havuz.items():
        if (v["rapor"].get("n") or 0) < 100:        # kucuk n merkez olamaz (sablon: n<100'e guvenme)
            continue
        ks = komsular(v["params"], havuz) + [ad]
        f = [havuz[k]["rapor"].get("fark_w3") for k in ks]
        f = [x for x in f if x is not None]
        if f and np.mean(f) > bs:
            bs, best = float(np.mean(f)), ad
    return best, bs


def aday_mi(ad, ham):
    v = ham[ad]; o = v["rapor"]; c = v["params"]
    havuz = {k: x for k, x in ham.items() if x["bolum"] == "train"}
    nb = komsular(c, havuz)
    ayni = sum(1 for k in nb if (havuz[k]["rapor"].get("fark_w3") or -1) > 0 and (havuz[k]["rapor"].get("wins3") or -1) > 0)
    kos1 = o.get("wins3") is not None and o["wins3"] > 0
    kos2 = o.get("fark_w3") is not None and o["fark_w3"] > 0 and o.get("fark_ga95") and o["fark_ga95"][0] > 0
    kos3 = o.get("top1_pay") is not None and o["top1_pay"] < 0.5
    kos4 = len(nb) > 0 and ayni > len(nb) / 2
    return dict(w3_poz=bool(kos1), plasebo_ustu_ga=bool(kos2), top1_ok=bool(kos3),
                komsu=f"{ayni}/{len(nb)}", komsu_ok=bool(kos4), n_ok=o.get("n", 0) >= 100,
                aday=bool(kos1 and kos2 and kos3 and kos4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", default="duman")
    ap.add_argument("--coin", type=int, default=0)
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--tf", default="4h,1d")
    ap.add_argument("--adaylar", default="")
    a = ap.parse_args()
    tfs = a.tf.split(",")
    ham = ham_oku()
    if a.asama == "duman":
        cfgs = []
        for tf in tfs:
            cfgs.append(dict(tf=tf, ema=[20, 50], Y=Y_GRID[tf][1], P=10, R=R_LOOK, cikis="tp3", filtre="f0"))
            cfgs.append(dict(tf=tf, ema=[20, 50], Y=Y_GRID[tf][1], P=10, R=R_LOOK, cikis="ema50", filtre="f2"))
        calistir(cfgs, "train", a.coin or 10, a.isci, f"duman testi {a.coin or 10} coin", prefix="duman_")
    elif a.asama == "tarama":
        cfgs = [c for tf in tfs for c in tarama_cfgs(tf) if ad_of(c) not in ham]
        print(f"[tarama] {len(cfgs)} konfig TRAIN (zaten yapilanlar atlandi)")
        calistir(cfgs, "train", a.coin, a.isci, "asama1 tarama f0+tp3 (TEKRAR: iptal edilen ilk kosunun hatali ay-blok GA kayitlarinin yerine gecer)", ham=ham)
    elif a.asama == "bolge":
        cfgs = []
        for tf in tfs:
            merkez, sk = en_iyi_bolge(tf, ham)
            mc = ham[merkez]["params"]
            print(f"[bolge] {tf} merkez={merkez} komsu-ort fark_w3={sk:+.4f}")
            for cik in ("tp2", "tp3", "ema50", "ema20"):
                for f in ("f0", "f1", "f2"):
                    k = dict(mc, cikis=cik, filtre=f)
                    if ad_of(k) not in ham:
                        cfgs.append(k)
        print(f"[bolge] {len(cfgs)} konfig TRAIN")
        calistir(cfgs, "train", a.coin, a.isci, "asama2 bolge cikis x filtre", ham=ham)
    elif a.asama == "komsu":      # asama2'de umut vaat eden konfigin eksik komsulari
        cfgs = []
        for ad in a.adaylar.split(","):
            c = ham[ad]["params"]; tf = c["tf"]
            ei = [list(e) for e in EMA_CIFT].index(list(c["ema"])); yi = Y_GRID[tf].index(c["Y"]); pi = P_GRID.index(c["P"])
            for dim in range(3):
                for s in (-1, 1):
                    idx = [ei, yi, pi]; idx[dim] += s
                    if not 0 <= idx[dim] < [3, 3, 3][dim]:
                        continue
                    k = dict(c, ema=list(EMA_CIFT[idx[0]]), Y=Y_GRID[tf][idx[1]], P=P_GRID[idx[2]])
                    if ad_of(k) not in ham:
                        cfgs.append(k)
        print(f"[komsu] {len(cfgs)} konfig TRAIN")
        calistir(cfgs, "train", a.coin, a.isci, "asama3 komsu kontrolu", ham=ham)
    elif a.asama == "degerlendir":
        tr = {k: v for k, v in ham.items() if v["bolum"] == "train"}
        print(f"TRAIN konfig: {len(tr)}")
        for ad in sorted(tr, key=lambda k: -(tr[k]["rapor"].get("fark_w3") or -9)):
            print(satir(ad, tr[ad]["rapor"]), aday_mi(ad, ham))
    elif a.asama == "valid":
        if any(v["bolum"] == "valid" for v in ham.values()) or any(
                r.get("aile") == AILE and r.get("bolum") == "valid" for r in H.defter_oku()):
            sys.exit("VALID bu aile icin zaten kosuldu (TEK SEFER).")
        ads = [x for x in a.adaylar.split(",") if x][:3]
        cfgs = [ham[x]["params"] for x in ads]
        print(f"[valid] TEK SEFER: {ads}")
        calistir(cfgs, "valid", 0, a.isci, "VALID tek sefer", ham=ham)


if __name__ == "__main__":
    main()
