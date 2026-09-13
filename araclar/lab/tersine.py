"""AILE: tersine -- kesitsel kisa vadeli tersine donus (short-term reversal), portfoy bazli.

Hipotezler
  H1  Siralama: son L gun getirisi (L in {1,3,5,7}). En KOTU N coin LONG (+0.5 toplam), en IYI N coin
      SHORT (-0.5 toplam), esit agirlik (gross 1, net 0). N in {5,10}. Yeniden dengeleme: gunluk (rb=1)
      veya haftalik (rb=7; W haftada bir guncellenir, arada ayni hedef agirlik tutulur -> sim_portfoy
      arada kucuk surukleme islemleri yapar).
  H2  Sadece long bacak (en kotu N, toplam +1) ve FARK portfoyu: en kotu N (+0.5) - evren esit agirlik (-0.5).
  H3  Vol-olcekli sinyal: getiri_L / vol30 ile siralama (L/S).
  H4  BTC MA50 rejim ayrimi: getiri gunlerini BTC kapanis > MA50 (t kapanisinda bilinen) ustu/alti diye ayir.

Evren: her t gunu, son 30 gun ortalama V (V zaten USDT quote hacmi) ile ilk U (U in {30,50}); coin gecmisi
>= 120 gun (gunluk_panel(min_gun=120) coini ancak 120. gunden sonra panele sokar); o gun fiyat mevcut.
Evren dolmazsa (uygun coin < U) pozisyon yok.

Ileriye bakma: W[t] t kapanisindaki bilgiyle; sim_portfoy t+1 acilisinda uygular, getiri O[t+1]->O[t+2].

MALIYET: sim_portfoy(maliyet=0.0007, funding=0). Short bacak funding ELLE:
  g_net[t] = g[t] - 0.0003 * S[t],  S[t] = sum(|w| : w<0), sim_portfoy ile ayni w (W[t], O[t+1] nan olanlar 0)
  -> funding sadece short notional'a uygulanir.
Duyarlilik: maliyet 0.0015 ve 0.003 (sim_portfoy yeniden kosulur, funding ayni sekilde dusulur).

PLASEBO: ayni evren, ayni N, ayni dengeleme takvimi; long ve short kumesi evrenden RASTGELE (50 seed).

HAYATTA-KALMA: data1d_all bugunun vadeli listesi. Olcum: her dengeleme gununde en cok dusen %10 dilimde
(ve karsilastirma icin en cok yukselen %10), sonraki 30 gunde verisi biten (panel sonundan >30 gun once
son bari olan) coin sayisi.

Kullanim:
  python tersine.py --faz train                      # tum grid + plasebo + maliyet duyarliligi, SADECE train
  python tersine.py --faz valid --adaylar A,B,C      # TEK sefer, en fazla 3
"""
import argparse, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "tersine"
MIN_HIST = 120
ADV_WIN = 30
VOL_WIN = 30
MALIYET = 0.0007
FUNDING = 0.0003
N_PLASEBO = 50
CACHE = os.path.join(H.SCRATCH, "_tersine_panel.pkl")   # harness.gunluk_panel ciktisinin turevi (holdout kesik)

_P = None


def _roll_mean(X, w):
    Z = np.nan_to_num(X, nan=0.0)
    cs = np.cumsum(Z, axis=0)
    out = np.full(Z.shape, np.nan)
    out[w - 1:] = (cs[w - 1:] - np.vstack([np.zeros((1, Z.shape[1])), cs[:-w]])) / w
    return out


def onhesap():
    global _P
    if _P is not None:
        return _P
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            _P = pickle.load(f)
        return _P
    syms, gun, O, C, V = H.gunluk_panel(min_gun=MIN_HIST)
    n, m = C.shape
    gecerli = ~np.isnan(C)
    cntv = _roll_mean(gecerli.astype(float), ADV_WIN)
    adv = _roll_mean(np.where(gecerli, V, np.nan), ADV_WIN)
    adv[~(cntv >= 0.999)] = np.nan                       # pencerede eksik gun -> uygun degil
    Cp = np.vstack([np.full((1, m), np.nan), C[:-1]])
    r1 = C / Cp - 1.0
    ok1 = np.isfinite(r1)
    R = np.nan_to_num(r1)
    c = _roll_mean(ok1.astype(float), VOL_WIN)
    m1 = _roll_mean(R, VOL_WIN); m2 = _roll_mean(R ** 2, VOL_WIN)
    vol = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)); vol[~(c >= 0.999)] = np.nan
    jb = syms.index("BTCUSDT")
    cb = C[:, jb]; mab = _roll_mean(cb.reshape(-1, 1), 50).ravel()
    cntb = _roll_mean(np.isfinite(cb).reshape(-1, 1).astype(float), 50).ravel()
    rejim = np.where((cb > mab) & (cntb >= 0.999), 1, 0)
    last = np.array([np.flatnonzero(gecerli[:, j])[-1] for j in range(m)])
    _P = dict(syms=syms, gun=np.asarray(gun), O=O, C=C, gecerli=gecerli, adv=adv, vol=vol,
              rejim=rejim, jb=jb, n=n, m=m, last=last)
    with open(CACHE, "wb") as f:
        pickle.dump(_P, f)
    return _P


def evren(P, t, U):
    elig = P["gecerli"][t] & np.isfinite(P["adv"][t]) & (P["adv"][t] > 0)
    ei = np.flatnonzero(elig)
    if len(ei) < U:
        return None
    return ei[np.argsort(-P["adv"][t, ei])[:U]]


def build_W(P, L=3, N=5, U=30, rb=1, mod="ls", sig="ret", seed=None, surv=None):
    """mod: ls | long | fark | ew | btc. seed != None -> plasebo (rastgele secim)."""
    n, m, C = P["n"], P["m"], P["C"]
    W = np.zeros((n, m))
    if mod == "btc":
        W[50:, P["jb"]] = 1.0
        return W
    rng = np.random.default_rng(seed) if seed is not None else None
    t0 = max(ADV_WIN, VOL_WIN, 8) + 1
    baz = np.zeros(m)
    for t in range(t0, n):
        if (t - t0) % rb == 0:
            baz = np.zeros(m)
            ev = evren(P, t, U)
            if ev is not None:
                if mod == "ew":
                    baz[ev] = 1.0 / len(ev)
                else:
                    x = C[t, ev] / C[t - L, ev] - 1.0
                    if sig == "vol":
                        x = x / P["vol"][t, ev]
                    ok = np.isfinite(x)
                    ev2, x = ev[ok], x[ok]
                    if len(ev2) >= max(2 * N, int(0.8 * U)):
                        srt = np.argsort(x, kind="stable")
                        if rng is not None:
                            perm = rng.permutation(len(ev2)); lo, hi = ev2[perm[:N]], ev2[perm[N:2 * N]]
                        else:
                            lo, hi = ev2[srt[:N]], ev2[srt[-N:]]
                        if surv is not None and rng is None and P["gun"][t] < H.T_VALID:   # sadece TRAIN tarihleri
                            k = max(1, int(round(0.1 * len(ev2))))
                            sinir = n - 31
                            for tag, grp in (("alt", ev2[srt[:k]]), ("ust", ev2[srt[-k:]])):
                                lst = P["last"][grp]
                                surv[tag + "_dilim"] += len(grp)
                                surv[tag + "_biten"] += int(((lst > t) & (lst <= t + 30) & (lst < sinir)).sum())
                        if mod == "rankw":
                            # EK (post-hoc, TANI'dan): tum evren rank-agirlikli L/S. skor = -getiri (dusen yuksek).
                            sk = rng.random(len(ev2)) if rng is not None else -x
                            rk = np.empty(len(ev2)); rk[np.argsort(sk, kind="stable")] = np.arange(len(ev2))
                            d = rk - rk.mean()
                            if np.abs(d).sum() > 0:
                                baz[ev2] = d / np.abs(d).sum()          # gross 1, net 0, long bacak +0.5
                        elif mod == "ls":
                            baz[lo] = 0.5 / N; baz[hi] = -0.5 / N
                        elif mod == "long":
                            baz[lo] = 1.0 / N
                        elif mod == "fark":
                            baz[ev] = -0.5 / len(ev); baz[lo] += 0.5 / N
        W[t] = baz
    return W


def sim(P, W, maliyet):
    """sim_portfoy + short-notional funding (elle) + turnover (sim_portfoy ile ayni surukleme)."""
    O = P["O"]; n = O.shape[0]
    eq, g = H.sim_portfoy(O, P["C"], W, maliyet=maliyet, funding=0.0)
    S = np.zeros(len(g)); to = np.zeros(len(g)); w_prev = np.zeros(P["m"])
    for t in range(n - 2):
        w = np.nan_to_num(W[t]).copy(); w[np.isnan(O[t + 1])] = 0
        S[t] = -w[w < 0].sum()
        to[t] = np.abs(w - w_prev).sum()
        r = np.clip(np.nan_to_num((O[t + 2] - O[t + 1]) / O[t + 1]), -0.95, 5.0)
        x = g[t]
        w_prev = w * (1 + r) / (1 + x) if x > -1 else w
    return g - FUNDING * S, to, S


def rap(P, g, to, S, bolum):
    gun = P["gun"][:len(g)]
    a, b = H.BOLUM[bolum]; msk = (gun >= a) & (gun < b)
    r = H.portfoy_rapor(None, g, P["gun"], bolum=bolum, kaynak="1d")
    r["ciro_yil"] = round(float(to[msk].mean() * 365), 1)
    r["short_notional_ort"] = round(float(S[msk].mean()), 3)
    r["aktif_gun_orani"] = round(float((np.abs(g[msk]) > 0).mean()), 3)
    reg = P["rejim"][:len(g)][msk]; gg = g[msk]
    for ad, sel in (("btc_ust", reg == 1), ("btc_alt", reg == 0)):
        x = gg[sel]
        r[ad] = dict(gun=int(sel.sum()), ort_gun_bps=round(float(x.mean() * 1e4), 2) if len(x) else None,
                     yillik_basit=round(float(x.mean() * 365), 3) if len(x) else None,
                     sharpe=round(float(x.mean() / x.std() * np.sqrt(365)), 2) if len(x) > 30 and x.std() > 0 else None)
    return r


def sharpe_bolum(P, g, bolum):
    gun = P["gun"][:len(g)]; a, b = H.BOLUM[bolum]; x = g[(gun >= a) & (gun < b)]
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


# ------------------------------------------------------------ worker'lar (modul seviyesi)
def w_kosu(args):
    ad, kw, bolum = args
    P = onhesap()
    surv = dict(alt_dilim=0, alt_biten=0, ust_dilim=0, ust_biten=0) if kw.get("mod") in ("ls", "long", "fark") else None
    W = build_W(P, **kw, surv=surv)
    out = {}
    for c in (MALIYET, 0.0015, 0.003):
        g, to, S = sim(P, W, c)
        out[c] = rap(P, g, to, S, bolum)
    return ad, out, surv


def w_plasebo(args):
    ad, kw, seed, bolum = args
    P = onhesap()
    W = build_W(P, **kw, seed=seed)
    g, to, S = sim(P, W, MALIYET)
    return ad, seed, sharpe_bolum(P, g, bolum)


def grid():
    cfg = {}
    for U in (30, 50):
        for rb in (1, 7):
            for N in (5, 10):
                for L in (1, 3, 5, 7):
                    base = dict(L=L, N=N, U=U, rb=rb)
                    cfg[f"LS_L{L}_N{N}_U{U}_rb{rb}"] = dict(base, mod="ls", sig="ret")
                    cfg[f"LSVOL_L{L}_N{N}_U{U}_rb{rb}"] = dict(base, mod="ls", sig="vol")
                    cfg[f"LONG_L{L}_N{N}_U{U}_rb{rb}"] = dict(base, mod="long", sig="ret")
                    cfg[f"FARK_L{L}_N{N}_U{U}_rb{rb}"] = dict(base, mod="fark", sig="ret")
    return cfg


def ek_grid():
    """EK (post-hoc): TANI'da rank IC train'de her yil pozitif (0.03-0.04) ama uc N=5/10 portfoy brut ~0/negatif.
    Kenar kuyruklarda degil siralamanin genelinde -> tum evren rank-agirlikli L/S."""
    cfg = {}
    for U in (30, 50):
        for rb in (1, 7):
            for L in (1, 3, 5, 7):
                cfg[f"RANKW_L{L}_Nall_U{U}_rb{rb}"] = dict(L=L, N=1, U=U, rb=rb, mod="rankw", sig="ret")
    return cfg


def kiyaslar():
    return {"KIYAS_btc": dict(mod="btc"), "KIYAS_ew30": dict(mod="ew", U=30), "KIYAS_ew50": dict(mod="ew", U=50)}


def yil_poz(r):
    y = r.get("yillar", {})
    return sum(1 for k in range(2019, 2025) if y.get(k, y.get(str(k), -1)) > 0)


def komsular(ad, sh):
    """ayni mod/U/rb: L +-1 adim ve diger N'nin train Sharpe'lari"""
    p = ad.split("_"); mod = p[0]; L = int(p[1][1:]); Ntok = p[2][1:]; rest = "_".join(p[3:])
    Ls = [1, 3, 5, 7]; i = Ls.index(L)
    kom = {}
    for j in (i - 1, i + 1):
        if 0 <= j < 4:
            k = f"{mod}_L{Ls[j]}_N{Ntok}_{rest}"; kom[k] = sh.get(k)
    if Ntok.isdigit():
        k = f"{mod}_L{L}_N{15 - int(Ntok)}_{rest}"; kom[k] = sh.get(k)
    else:   # RANKW: N yok -> diger evren U
        U = p[3]; k = f"{mod}_L{L}_N{Ntok}_{'U50' if U == 'U30' else 'U30'}_{p[4]}"; kom[k] = sh.get(k)
    return kom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="train", choices=["train", "ek", "valid"])
    ap.add_argument("--adaylar", default="")
    ap.add_argument("--isci", type=int, default=14)
    ap.add_argument("--plasebo", type=int, default=N_PLASEBO)
    a = ap.parse_args()
    t_bas = time.time()
    P = onhesap()
    print(f"panel {P['m']} coin {P['n']} gun, {time.time()-t_bas:.0f}s", flush=True)
    bolum = "train" if a.faz in ("train", "ek") else "valid"
    cfg = dict(grid(), **kiyaslar()) if a.faz == "train" else (dict(ek_grid(), **kiyaslar()) if a.faz == "ek" else dict(grid(), **ek_grid(), **kiyaslar()))
    if a.faz == "valid":
        adl = [x for x in a.adaylar.split(",") if x]
        assert 1 <= len(adl) <= 3, "VALID: en fazla 3 aday"
        cfg = {k: cfg[k] for k in adl + list(kiyaslar())}
    sonuc = {}
    with ProcessPoolExecutor(a.isci) as ex:
        for ad, out, surv in ex.map(w_kosu, [(k, v, bolum) for k, v in cfg.items()]):
            sonuc[ad] = dict(params=cfg[ad], rapor=out, surv=surv)
            r = out[MALIYET]
            print(f"{ad:24s} sh={r.get('sharpe')} cagr={r.get('cagr')} dd={r.get('maxdd')} ciro={r.get('ciro_yil')} "
                  f"sh15={out[0.0015].get('sharpe')} sh30={out[0.003].get('sharpe')} yil={r.get('yillar')}", flush=True)
        print(f"kosular bitti {time.time()-t_bas:.0f}s", flush=True)
        ps = [(k, v, s, bolum) for k, v in cfg.items() if not k.startswith("KIYAS") for s in range(a.plasebo)]
        dag = {}
        for ad, seed, sh in ex.map(w_plasebo, ps, chunksize=8):
            dag.setdefault(ad, []).append(sh)
        print(f"plasebo bitti {time.time()-t_bas:.0f}s", flush=True)
    for ad, d in dag.items():
        sh = sonuc[ad]["rapor"][MALIYET].get("sharpe", 0)
        arr = np.array(d)
        sonuc[ad]["plasebo"] = dict(n=len(arr), ort=round(float(arr.mean()), 3), std=round(float(arr.std()), 3),
                                    p95=round(float(np.percentile(arr, 95)), 3), p99=round(float(np.percentile(arr, 99)), 3),
                                    maks=round(float(arr.max()), 3), yuzdelik=round(float((arr < sh).mean() * 100), 1))
    for ad, s in sonuc.items():
        for c, r in s["rapor"].items():
            rr = dict(r)
            if c == MALIYET and "plasebo" in s:
                rr["plasebo"] = s["plasebo"]
            if c == MALIYET and s.get("surv"):
                rr["surv"] = s["surv"]
            H.kaydet(AILE, ad if c == MALIYET else f"{ad}__maliyet{c}",
                     dict(s["params"], maliyet=c, funding_short=FUNDING, min_hist=MIN_HIST, n_plasebo=a.plasebo),
                     bolum, rr, not_="funding sadece short notional'a elle dusuldu; sim_portfoy funding=0")
    if a.faz in ("train", "ek"):
        sh = {k: v["rapor"][MALIYET].get("sharpe") for k, v in sonuc.items()}
        for ad, s in sonuc.items():
            if ad.startswith("KIYAS"):
                continue
            r = s["rapor"][MALIYET]; kom = komsular(ad, sh); s0 = r.get("sharpe", 0)
            ayni = all(v is not None and (v > 0) == (s0 > 0) for v in kom.values())
            s["olcut"] = dict(sharpe_gt1=s0 > 1.0, plasebo_gt99=s["plasebo"]["yuzdelik"] > 99,
                              maliyet15_gt05=s["rapor"][0.0015].get("sharpe", 0) > 0.5,
                              komsu_ayni_yon=ayni, yil_poz_2019_24=yil_poz(r), yil_4_6=yil_poz(r) >= 4, komsular=kom)
            s["olcut"]["hepsi"] = all(s["olcut"][k] for k in ("sharpe_gt1", "plasebo_gt99", "maliyet15_gt05", "komsu_ayni_yon", "yil_4_6"))
    with open(os.path.join(LAB, f"_tersine_{a.faz}.json"), "w", encoding="utf-8") as f:
        json.dump({k: dict(v, rapor={str(c): r for c, r in v["rapor"].items()}) for k, v in sonuc.items()}, f, indent=1, default=str)
    print(f"bitti {time.time()-t_bas:.0f}s", flush=True)


if __name__ == "__main__":
    main()
