"""funding - Binance USDT-M perpetual funding oraninin iki kullanimi (kaynak=1d, portfoy bazli).

H1 CARRY (delta-notr): long spot + short perp esit notional.
   Gunluk PnL / notional = funding(gun) + (spot getiri - perp getiri).
   SERMAYE VARSAYIMI: sermayenin yarisi spot alimina, yarisi perp teminatina gider (perp 1x) ->
   notional = 0.5 * sermaye. Sermaye getirisi = 0.5 * notional getirisi. Kaldirac yok.
   Maliyet (bir birim notional acma VEYA kapama): spot taker 0.10% + perp taker 0.05% + kayma 0.02%*2 bacak
   = 0.19% notional -> sim_portfoy'da agirlik sermaye cinsinden (toplam <= 0.5), maliyet=0.0019 / |dW|.
   Teminat faizi yok, spot USDT faizi yok (muhafazakar).
   a) hep acik esit agirlik  b) esikli histerezis (7g ort yillik funding > E ac, < E/2 kapa)
   c) kesitsel haftalik top-K (son 7g funding)
H2 FUNDING SINYALI (yonlu): 3g/7g ort funding'in kendi 90g dagilimindaki yuzdeligi -> ileri 1/3/7g spot getiri.

Zamanlama: karar t gun KAPANISINDA (funding t gunune kadar olan odemeler, spot/perp t kapanisi),
uygulama t+1 acilisi (harness.sim_portfoy konvansiyonu). Funding odemesi zaman damgasi (gun_d 00:00, gun_d+1 00:00]
araligindaysa gun_d'ye yazilir (pozisyon gun_d acilisindan gun_d+1 acilisina tutulur).

HOLDOUT: indirilen 2026 verisi analizden ONCE kesilir (ts < 2026-01-01). LAB_HOLDOUT ayarlanmaz.

Kullanim:
  PYTHONIOENCODING=utf-8 python funding.py --faz indir
  PYTHONIOENCODING=utf-8 python funding.py --faz h1       # TRAIN
  PYTHONIOENCODING=utf-8 python funding.py --faz h2       # TRAIN
  PYTHONIOENCODING=utf-8 python funding.py --faz valid    # adaylar (en fazla 3) VALID tek sefer
  PYTHONIOENCODING=utf-8 python funding.py --faz yaz
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "funding"
DDIR = os.path.join(H.SCRATCH, "data_funding")
GUN = 86400_000
T_FUND0 = int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
CEKIRDEK = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT",
            "LINKUSDT", "BCHUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"]
TOPN = 20
MAL_CARRY = 0.0010 + 0.0005 + 2 * 0.0002      # 0.19% notional, tek yon (ac veya kapa)
NOTIONAL_SERMAYE = 0.5


def ts2d(ms):
    return datetime.fromtimestamp(ms / 1e3, timezone.utc).strftime("%Y-%m-%d")


# =================================================================== INDIRME
def _get(url, deneme=8):
    for k in range(deneme):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 418):
                w = int(e.headers.get("Retry-After", 60) or 60)
                print(f"  {e.code} bekle {w}s"); time.sleep(min(w, 120)); continue
            if e.code == 400:
                return None          # sembol yok
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return None


def indir_funding(sym):
    out = []; st = T_FUND0
    while True:
        d = _get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={st}&limit=1000")
        if not d:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in d]
        if len(d) < 1000:
            break
        st = int(d[-1]["fundingTime"]) + 1
    return np.array(out) if out else None


def indir_klines(sym):
    out = []; st = T_FUND0
    while True:
        d = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d&startTime={st}&limit=1500")
        if not d:
            break
        out += [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])] for x in d]
        if len(d) < 1500:
            break
        st = int(d[-1][0]) + GUN
    return np.array(out) if out else None


def aday_semboller(topk=25):
    """Spot panelinden (harness) her ay basi 90g ort dolar hacmi ilk-topk birlesimi (2019-06..2025-12)."""
    syms, gun, O, C, V = H.gunluk_panel(min_gun=1)
    dv = _roll_mean(np.nan_to_num(V), 90)
    dv[np.isnan(C)] = np.nan
    ay = np.array([_ay(g) for g in gun]); S = set(CEKIRDEK)
    t0 = np.searchsorted(gun, int(datetime(2019, 6, 1, tzinfo=timezone.utc).timestamp() * 1000))
    for t in range(max(t0, 1), len(gun)):
        if ay[t] != ay[t - 1]:
            x = np.where(np.isnan(dv[t - 1]), -np.inf, dv[t - 1])
            for j in np.argsort(-x)[:topk]:
                if np.isfinite(x[j]):
                    S.add(syms[j])
    return sorted(S)


def _tek_indir(sym):
    pf = os.path.join(DDIR, f"fund_{sym}.npy"); pk = os.path.join(DDIR, f"perp_{sym}.npy")
    if os.path.exists(pf) and os.path.exists(pk):
        return sym, "var"
    adaylar = [sym] if sym.startswith("1000") else [sym, "1000" + sym]
    for ps in adaylar:
        f = indir_funding(ps)
        if f is not None and len(f):
            k = indir_klines(ps)
            if k is None:
                continue
            if ps != sym:          # 1000X perp -> fiyat olcegi farkli, getiri ayni
                k[:, 1:5] /= 1000.0
            np.save(pf, f); np.save(pk, k)
            return sym, f"ok({ps}) f={len(f)} k={len(k)} bas={ts2d(f[0,0])}"
    return sym, "funding yok"


def faz_indir():
    os.makedirs(DDIR, exist_ok=True)
    S = aday_semboller()
    print("aday sembol:", len(S))
    log = {}
    with ThreadPoolExecutor(8) as ex:
        for sym, m in ex.map(_tek_indir, S):
            log[sym] = m; print(" ", sym, m)
    json.dump(log, open(os.path.join(DDIR, "_log.json"), "w"), indent=1)


# =================================================================== YARDIMCI
def _roll_mean(A, N):
    """NaN-farkindali: pencerenin N degerinin hepsi gecerliyse ortalama, degilse NaN"""
    A = np.asarray(A, float); z = np.zeros((1,) + A.shape[1:])
    cs = np.cumsum(np.vstack([z, np.nan_to_num(A)]), 0); cc = np.cumsum(np.vstack([z, np.isfinite(A)]), 0)
    out = np.full_like(A, np.nan)
    s = cs[N:] - cs[:-N]; c = cc[N:] - cc[:-N]
    out[N - 1:] = np.where(c == N, s / N, np.nan)
    return out


def _ay(ms):
    d = datetime.fromtimestamp(ms / 1e3, timezone.utc); return d.year * 12 + d.month


# =================================================================== PANEL
T0 = int(datetime(2019, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)   # spot hacim isinmasi icin
_P = {}


def panel():
    """Gun izgarasi T0..2025-12-31. HER dizi analizden once ts < 2026-01-01 kesilir."""
    if _P:
        return _P
    syms = sorted(f[5:-4] for f in os.listdir(DDIR) if f.startswith("fund_") and
                  os.path.exists(os.path.join(DDIR, "perp_" + f[5:])))
    n = int((H.T_HOLD - T0) // GUN); m = len(syms)
    gun = T0 + GUN * np.arange(n, dtype=np.int64)
    Os, Cs, Vs, Op, Fd = (np.full((n, m), np.nan) for _ in range(5))
    Fsum = np.zeros((n, m)); Fcnt = np.zeros((n, m))
    for j, s in enumerate(syms):
        try:
            a = H.veri_1d(s)
        except Exception:
            continue
        a = a[(a[:, 0] >= T0) & (a[:, 0] < H.T_HOLD)]
        ix = ((a[:, 0] - T0) // GUN).astype(int)
        Os[ix, j] = a[:, 1]; Cs[ix, j] = a[:, 4]; Vs[ix, j] = a[:, 5]
        k = np.load(os.path.join(DDIR, f"perp_{s}.npy")); k = k[(k[:, 0] >= T0) & (k[:, 0] < H.T_HOLD)]
        # olu perp piyasasi (delist sonrasi donmus mum, orn FTT 2023: fiyat 1.59 sabit, hacim 0) -> gecersiz
        k = k[(k[:, 5] > 0) & (k[:, 2] > k[:, 3])]
        ix = ((k[:, 0] - T0) // GUN).astype(int); Op[ix, j] = k[:, 1]
        f = np.load(os.path.join(DDIR, f"fund_{s}.npy")); f = f[f[:, 0] < H.T_HOLD]      # HOLDOUT KESIMI
        d = ((f[:, 0] - 60_000 - T0) // GUN).astype(int); ok = (d >= 0) & (d < n)
        np.add.at(Fsum[:, j], d[ok], f[ok, 1]); np.add.at(Fcnt[:, j], d[ok], 1)
        live = np.flatnonzero(Fcnt[:, j] > 0)
        if len(live):
            seg = slice(live[0], live[-1] + 1)
            Fd[seg, j] = np.where(Fcnt[seg, j] > 0, Fsum[seg, j], np.nan)
    assert gun[-1] < H.T_HOLD
    with np.errstate(invalid="ignore", divide="ignore"):
        uyumsuz = np.abs(Op / Os - 1) > 0.30        # spot/perp farkli varlik veya bozuk veri (gercek FTX gunleri <%10)
    Op[uyumsuz] = np.nan
    # carry getirisi (notional basina), gun d acilisi -> d+1 acilisi
    rs = np.full((n, m), np.nan); rp = np.full((n, m), np.nan)
    rs[:-1] = Os[1:] / Os[:-1] - 1; rp[:-1] = Op[1:] / Op[:-1] - 1
    x = Fd + rs - rp
    I = np.full((n, m), np.nan)
    for j in range(m):
        L = 1.0
        for d in range(n - 1):
            if np.isfinite(x[d, j]):
                L *= 1 + x[d, j]; I[d + 1, j] = L
    f7 = _roll_mean(Fd, 7) * 365; f3 = _roll_mean(Fd, 3) * 365
    dv = _roll_mean(np.nan_to_num(Vs), 90); dv[np.isnan(Cs)] = np.nan
    _P.update(syms=syms, gun=gun, Os=Os, Cs=Cs, Op=Op, Fd=Fd, rs=rs, rp=rp, x=x, I=I, f7=f7, f3=f3, dv=dv,
              ay=np.array([_ay(g) for g in gun]))
    return _P


_EV = {}
def evren(tip="genis"):
    """t kapanisinda bilinen evren. genis = CEKIRDEK U aylik top-20 (ay basi, bir onceki gunun 90g ort spot
    dolar hacmi; sadece o gun funding verisi olan semboller). Ileriye bakma yok."""
    if tip in _EV:
        return _EV[tip]
    P = panel(); syms = P["syms"]; n, m = P["x"].shape
    ok = np.isfinite(P["I"]) & np.isfinite(P["f7"]) & np.isfinite(P["Os"]) & np.isfinite(P["Op"])
    core = np.array([s in CEKIRDEK for s in syms])
    if tip == "cekirdek":
        M = ok & core[None, :]
    else:
        M = np.zeros((n, m), bool); sec = np.zeros(m, bool); P["top20_tarihce"] = []
        for t in range(1, n):
            if P["ay"][t] != P["ay"][t - 1]:
                src = t - 1
                sk = np.where(ok[src] & np.isfinite(P["dv"][src]), P["dv"][src], -np.inf)
                o = [j for j in np.argsort(-sk)[:TOPN] if np.isfinite(sk[j])]
                sec = np.zeros(m, bool); sec[o] = True
                P["top20_tarihce"].append([ts2d(P["gun"][t])[:7], [syms[j].replace("USDT", "") for j in o]])
            M[t] = sec | core
        M &= ok
    _EV[tip] = M
    return M


def _pazartesi(gun):
    return ((np.asarray(gun) // GUN) % 7) == 4


# =================================================================== H1 AGIRLIKLAR (notional/sermaye, toplam <= 0.5)
def w_hepacik(ev):
    U = evren(ev); K = U.sum(1, keepdims=True)
    return np.where(U, NOTIONAL_SERMAYE / np.maximum(K, 1), 0.0)


def w_esik(ev, E, mod="sabit_slot", cap=0.10):
    P = panel(); U = evren(ev); f7 = P["f7"]; n, m = U.shape
    on = np.zeros(m, bool); A = np.zeros((n, m), bool)
    for t in range(n):
        on = np.where(~U[t], False, np.where(f7[t] > E, True, np.where(f7[t] < E / 2, False, on)))
        A[t] = on
    if mod == "sabit_slot":
        K = U.sum(1, keepdims=True)
        return np.where(A, NOTIONAL_SERMAYE / np.maximum(K, 1), 0.0)
    K = A.sum(1, keepdims=True)
    return np.where(A, np.minimum(NOTIONAL_SERMAYE / np.maximum(K, 1), cap), 0.0)


def w_kesitsel(ev, K, rng=None):
    P = panel(); U = evren(ev); f7 = P["f7"]; n, m = U.shape
    pzt = _pazartesi(P["gun"]); W = np.zeros((n, m)); cur = np.zeros(m)
    for t in range(n):
        if pzt[t]:
            cand = np.flatnonzero(U[t]); cur = np.zeros(m)
            if len(cand):
                if rng is None:
                    sel = cand[np.argsort(-f7[t, cand])[:K]]
                else:
                    sel = rng.choice(cand, min(K, len(cand)), replace=False)
                cur[sel] = NOTIONAL_SERMAYE / K
        W[t] = np.where(U[t], cur, 0.0)
    return W


# =================================================================== OLCUM
OLAYLAR = {"2021-05_cokus": ("2021-05-17", "2021-05-24"), "2022-06_3AC_celsius": ("2022-06-11", "2022-06-19"),
           "2022-11_FTX": ("2022-11-06", "2022-11-14")}


def _ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def calistir(W, maliyet=MAL_CARRY):
    P = panel()
    eq, g = H.sim_portfoy(P["I"], P["I"], W, maliyet=maliyet)
    n = len(g)
    Wa = np.nan_to_num(W[:n]).copy(); Wa[np.isnan(P["I"][1:n + 1])] = 0
    fk = (Wa * np.nan_to_num(P["Fd"][1:n + 1])).sum(1)
    bk = (Wa * np.nan_to_num(P["rs"][1:n + 1] - P["rp"][1:n + 1])).sum(1)
    return dict(eq=eq, g=g, gun=P["gun"][:n], fk=fk, bk=bk, mk=g - fk - bk, W=Wa)


_BTC = {}
def btc_g():
    if not _BTC:
        P = panel(); j = P["syms"].index("BTCUSDT"); W = np.zeros_like(P["Os"]); W[:, j] = 1.0
        eq, g = H.sim_portfoy(P["Os"], P["Cs"], W, maliyet=0.0007); _BTC["g"] = g
    return _BTC["g"]


def olc(R, bolum):
    g, gun = R["g"], R["gun"]; a, b = H.BOLUM[bolum]
    ms = (gun >= a) & (gun < b)
    ilk = np.flatnonzero(np.abs(R["W"]).sum(1) > 0)        # analiz: ilk pozisyondan itibaren
    if len(ilk):
        ms &= np.arange(len(g)) >= ilk[0]
    rep = H.portfoy_rapor(R["eq"], g[ms], gun[ms])
    if rep.get("gun", 0) < 30:
        return rep
    gg, gd = g[ms], gun[ms]; yil = len(gg) / 365
    rep["vol"] = round(float(gg.std() * np.sqrt(365)), 4)
    ay = {}
    for t, v in zip(gd, gg):
        ay.setdefault(ts2d(t)[:7], []).append(v)
    aylik = {k: float(np.prod(1 + np.array(v)) - 1) for k, v in ay.items()}
    kotu = min(aylik, key=aylik.get)
    rep["en_kotu_ay"] = [kotu, round(aylik[kotu], 4)]
    rep["negatif_ay_orani"] = round(float(np.mean([v < 0 for v in aylik.values()])), 3)
    rep["katki_yillik"] = dict(funding=round(float(R["fk"][ms].sum() / yil), 4), baz=round(float(R["bk"][ms].sum() / yil), 4),
                               maliyet=round(float(R["mk"][ms].sum() / yil), 4))
    rep["ort_notional"] = round(float(np.abs(R["W"][ms]).sum(1).mean()), 3)
    rep["ciro_yil"] = round(float(np.abs(np.diff(R["W"][ms], axis=0)).sum() / yil), 2)
    neg = R["fk"][ms] < 0
    rep["funding_negatif_gunler"] = dict(gun=int(neg.sum()), toplam_getiri=round(float(gg[neg].sum()), 4),
                                         funding_kaybi=round(float(R["fk"][ms][neg].sum()), 4),
                                         baz=round(float(R["bk"][ms][neg].sum()), 4))
    bg = btc_g()[:len(g)][ms]
    rep["btc_korelasyon"] = round(float(np.corrcoef(gg, bg)[0, 1]), 3) if gg.std() > 0 else None
    o = np.argsort(gg)[:5]
    rep["en_kotu_5_gun"] = [[ts2d(gd[i] + GUN), round(float(gg[i]), 4), round(float(R["fk"][ms][i]), 4),
                             round(float(R["bk"][ms][i]), 4)] for i in o]     # [getiri gunu, toplam, funding, baz]
    return rep


def olaylar(R):
    """kuyruk olaylari: pencere icindeki her gun [getiri gunu, toplam, funding, baz] + en kotu coin-baz"""
    P = panel(); out = {}
    for ad, (s, e) in OLAYLAR.items():
        a, b = _ms(s) - GUN, _ms(e) - GUN      # g[t] = gun t+1 getirisi
        idx = np.flatnonzero((R["gun"] >= a) & (R["gun"] <= b))
        gunler = [[ts2d(R["gun"][i] + GUN), round(float(R["g"][i]), 5), round(float(R["fk"][i]), 5),
                   round(float(R["bk"][i]), 5)] for i in idx]
        coin = []
        for i in idx:
            bz = R["W"][i] * np.nan_to_num(P["rs"][i + 1] - P["rp"][i + 1])
            for j in np.argsort(bz)[:2]:
                if bz[j] < -1e-5:
                    coin.append([ts2d(R["gun"][i] + GUN), P["syms"][j], round(float(P["rs"][i + 1, j] - P["rp"][i + 1, j]), 4),
                                 round(float(bz[j]), 5)])
        coin.sort(key=lambda r: r[3])
        out[ad] = dict(toplam=round(float(np.prod(1 + R["g"][idx]) - 1), 4), gunler=gunler,
                       en_kotu_coin_baz=coin[:5])        # [gun, coin, spot-perp getiri farki (notional), sermaye etkisi]
    return out


def aday_mi(r):
    return bool((r.get("sharpe") or 0) > 1.0 and (r.get("cagr") or 0) > 0.05 and (r.get("maxdd") or 1) < 0.15)


# =================================================================== H1 FAZ
def h1_konfig():
    K = [("H1a_hepacik__genis", dict(kural="a", evren="genis"), lambda: w_hepacik("genis")),
         ("H1a_hepacik__cekirdek", dict(kural="a", evren="cekirdek"), lambda: w_hepacik("cekirdek"))]
    for ev in ("genis", "cekirdek"):
        for E in (0.05, 0.10, 0.20, 0.30):
            K.append((f"H1b_esik{int(E*100)}__{ev}", dict(kural="b", evren=ev, E=E, mod="sabit_slot"),
                      (lambda ev=ev, E=E: w_esik(ev, E))))
    for E in (0.05, 0.10, 0.20, 0.30):
        K.append((f"H1b_esik{int(E*100)}_dagit__genis", dict(kural="b", evren="genis", E=E, mod="yeniden_dagit", cap=0.10),
                  (lambda E=E: w_esik("genis", E, "yeniden_dagit"))))
    for ev in ("genis", "cekirdek"):
        for k in (3, 5, 10):
            K.append((f"H1c_top{k}__{ev}", dict(kural="c", evren=ev, K=k), (lambda ev=ev, k=k: w_kesitsel(ev, k))))
    return K


def faz_h1(bolumler=("train",), sadece=None, kaydet=True):
    out = {}
    for ad, prm, fn in h1_konfig():
        if sadece and ad not in sadece:
            continue
        R = calistir(fn())
        prm = dict(prm, maliyet_notional=MAL_CARRY, notional_sermaye=NOTIONAL_SERMAYE)
        out[ad] = dict(params=prm)
        for bl in bolumler:
            r = olc(R, bl)
            out[ad][bl] = r
            if kaydet:
                H.kaydet(AILE, ad, prm, bl, r)
            print(f"{ad:<28} {bl:<5} cagr={r.get('cagr')} vol={r.get('vol')} S={r.get('sharpe')} dd={r.get('maxdd')} "
                  f"kotu_ay={r.get('en_kotu_ay')} katki={r.get('katki_yillik')} notl={r.get('ort_notional')} "
                  f"korel={r.get('btc_korelasyon')} aday={aday_mi(r)}")
        out[ad]["olaylar"] = olaylar(R) if "train" in bolumler else None
    return out


def plasebo_h1c(ev, K, bolumler=("train",), seeds=20, kaydet=True):
    res = {bl: [] for bl in bolumler}
    for sd in range(seeds):
        R = calistir(w_kesitsel(ev, K, rng=np.random.default_rng(sd)))
        for bl in bolumler:
            res[bl].append(olc(R, bl))
    out = {}
    for bl in bolumler:
        S = np.array([r["sharpe"] for r in res[bl]]); Cg = np.array([r["cagr"] for r in res[bl]])
        out[bl] = dict(sharpe_ort=round(float(S.mean()), 2), sharpe_p95=round(float(np.percentile(S, 95)), 2),
                       sharpe_max=round(float(S.max()), 2), cagr_ort=round(float(Cg.mean()), 4),
                       cagr_p95=round(float(np.percentile(Cg, 95)), 4),
                       maxdd_ort=round(float(np.mean([r["maxdd"] for r in res[bl]])), 3),
                       funding_katki_ort=round(float(np.mean([r["katki_yillik"]["funding"] for r in res[bl]])), 4))
        if kaydet:
            H.kaydet(AILE, f"PLASEBO_H1c_rastgele{K}__{ev}", dict(kural="c_plasebo", evren=ev, K=K, seeds=seeds), bl, out[bl])
        print(f"PLASEBO top{K} {ev} {bl}: {out[bl]}")
    return out


def btc_altut(bolumler=("train",)):
    P = panel(); g = btc_g(); gun = P["gun"][:len(g)]
    out = {}
    for bl in bolumler:
        a, b = H.BOLUM[bl]; ms = (gun >= a) & (gun < b) & (gun >= _ms("2019-09-10"))
        out[bl] = H.portfoy_rapor(None, g[ms], gun[ms]); out[bl]["vol"] = round(float(g[ms].std() * np.sqrt(365)), 3)
    return out


def coin_ozet(bolum="train"):
    """coin bazinda (genis evrende iken) yillik ort funding, funding<0 gun orani, toplam baz getirisi, en kotu gun baz"""
    P = panel(); a, b = H.BOLUM[bolum]; ms = (P["gun"] >= a) & (P["gun"] < b); U = evren("genis")
    out = {}
    for j, s in enumerate(P["syms"]):
        m = ms & U[:, j]
        if m.sum() < 120:
            continue
        f = P["Fd"][m, j]; bz = np.nan_to_num(P["rs"][m, j] - P["rp"][m, j])
        out[s] = dict(gun=int(m.sum()), funding_yillik=round(float(np.nanmean(f) * 365), 4),
                      funding_neg_oran=round(float(np.mean(f < 0)), 3), baz_toplam=round(float(bz.sum()), 4),
                      en_kotu_baz_gun=round(float(bz.min()), 4))
    return out


# =================================================================== H2 FUNDING SINYALI
_PCT = {}
def yuzdelik():
    """f3/f7'nin kendi son 90 gunluk (bugun dahil) dagilimindaki yuzdeligi; t kapanisinda bilinir."""
    if _PCT:
        return _PCT
    P = panel()
    for win, F in (("3g", P["f3"]), ("7g", P["f7"])):
        n, m = F.shape; pct = np.full((n, m), np.nan)
        for t in range(89, n):
            blk = F[t - 89:t + 1]; cur = F[t]
            cnt = np.isfinite(blk).sum(0)
            with np.errstate(invalid="ignore"):
                less = (blk < cur).sum(0) + 0.5 * ((blk == cur).sum(0) - 1)
            pct[t] = np.where((cnt >= 90) & np.isfinite(cur), less / np.maximum(cnt - 1, 1), np.nan)
        _PCT[win] = pct
    return _PCT


def _grup_istat(r, grp, key, nk):
    S = np.zeros((nk, 3)); N = np.zeros((nk, 3))
    np.add.at(S, (key, grp), r); np.add.at(N, (key, grp), 1)
    return S, N


def _fark(S, N):
    s = S.sum(0); c = N.sum(0)
    mu = s / np.maximum(c, 1)
    return mu[1] - mu[0], mu[2] - mu[0]


def h2_test(bolum="train", nboot=2000, testsay=12, ust=0.9, alt=0.1, kaydet=True):
    P = panel(); U = evren("genis"); pcts = yuzdelik(); Os = P["Os"]; gun = P["gun"]; n, m = Os.shape
    a, b = H.BOLUM[bolum]; tm = ((gun >= a) & (gun < b))[:, None]
    alfa = 0.05 / testsay; rng = np.random.default_rng(0)
    out = {}
    for win in ("3g", "7g"):
        for h in (1, 3, 7):
            fwd = np.full((n, m), np.nan)
            fwd[:n - 1 - h] = Os[1 + h:n] / Os[1:n - h] - 1          # t+1 acilis -> t+1+h acilis
            pct = pcts[win]
            ok = U & np.isfinite(pct) & np.isfinite(fwd) & tm
            ti, ji = np.nonzero(ok)
            r = fwd[ti, ji]; p = pct[ti, ji]
            lo_, hi_ = np.percentile(r, [0.5, 99.5]); rw = np.clip(r, lo_, hi_)        # winsorize
            dm = np.zeros(n); cn = np.zeros(n); np.add.at(dm, ti, rw); np.add.at(cn, ti, 1)
            rfe = rw - (dm / np.maximum(cn, 1))[ti]                                      # zaman sabit etkisi
            grp = np.where(p > ust, 1, np.where(p < alt, 2, 0))
            mon = P["ay"][ti]; mon = mon - mon.min(); nm = mon.max() + 1
            sonuc = {}
            for etiket, y in (("ham_w", rw), ("zamanFE", rfe)):
                Sc, Nc = _grup_istat(y, grp, ji, m); Sm, Nm = _grup_istat(y, grp, mon, nm)
                d_hi, d_lo = _fark(Sc, Nc)
                # coin bootstrap: sadece veri olan coinlerden cek
                cz = np.flatnonzero(Nc.sum(1) > 0)
                bc = np.array([_fark(Sc[cz[k]], Nc[cz[k]]) for k in rng.integers(0, len(cz), (nboot, len(cz)))])
                bm = np.array([_fark(Sm[k], Nm[k]) for k in rng.integers(0, nm, (nboot, nm))])
                q = [alfa / 2 * 100, (1 - alfa / 2) * 100]
                for yon, d, col in (("yuksek", d_hi, 0), ("dusuk", d_lo, 1)):
                    ci_c = np.percentile(bc[:, col], q); ci_m = np.percentile(bm[:, col], q)
                    p_c = float(min(1, 2 * min((bc[:, col] <= 0).mean(), (bc[:, col] >= 0).mean())))
                    p_m = float(min(1, 2 * min((bm[:, col] <= 0).mean(), (bm[:, col] >= 0).mean())))
                    sonuc[f"{etiket}_{yon}"] = dict(fark=round(float(d), 5), ga_coin=[round(float(v), 5) for v in ci_c],
                                                   ga_ay=[round(float(v), 5) for v in ci_m], p_coin=round(p_c, 4), p_ay=round(p_m, 4),
                                                   anlamli_bonf=bool((ci_c[0] > 0 or ci_c[1] < 0) and (ci_m[0] > 0 or ci_m[1] < 0)))
            sonuc["n"] = dict(orta=int((grp == 0).sum()), yuksek=int((grp == 1).sum()), dusuk=int((grp == 2).sum()),
                              coin=int(len(np.unique(ji))))
            sonuc["ham_ort"] = dict(orta=round(float(r[grp == 0].mean()), 5), yuksek=round(float(r[grp == 1].mean()), 5),
                                    dusuk=round(float(r[grp == 2].mean()), 5))
            key = f"{win}_h{h}"; out[key] = sonuc
            if kaydet:
                H.kaydet(AILE, f"H2_yuzdelik_{key}", dict(pencere=win, ufuk=h, ust=ust, alt=alt, bonferroni=testsay,
                                                          winsor=0.005, bootstrap=["coin", "ay-blok"]), bolum, sonuc)
            hw, lw = sonuc["ham_w_yuksek"], sonuc["ham_w_dusuk"]
            print(f"H2 {key} {bolum}: n={sonuc['n']} | yuksek-orta {hw['fark']:+.4f} GAc={hw['ga_coin']} GAay={hw['ga_ay']} "
                  f"sig={hw['anlamli_bonf']} | dusuk-orta {lw['fark']:+.4f} GAc={lw['ga_coin']} GAay={lw['ga_ay']} "
                  f"sig={lw['anlamli_bonf']} | FE yuk {sonuc['zamanFE_yuksek']['fark']:+.4f} sig={sonuc['zamanFE_yuksek']['anlamli_bonf']}"
                  f" FE dus {sonuc['zamanFE_dusuk']['fark']:+.4f} sig={sonuc['zamanFE_dusuk']['anlamli_bonf']}")
    return out


def h2_overlay(bolumler=("train",), win="7g", ust=0.9, seeds=20, kaydet=True):
    """tsmom2 s3_orij_bnb MA50+vt30 agirliklari x (funding yuzdeligi > ust ise 0.5). Plasebo: ayni bolumde, coin
    basina ayni sayida (pozisyonlu gunlerden) rastgele gun yarilanir."""
    import tsmom as T, tsmom2 as T2
    PT = T.panel(); b = PT["bas"]; cfg = T2.sepet_cfg("s3_orij_bnb", hedef=0.3)
    W = T2.agirliklar2(cfg)[b:]; gunT = PT["gun"][b:]
    P = panel(); pct = yuzdelik()[win]
    di = ((gunT - T0) // GUN).astype(int); vd = (di >= 0) & (di < len(P["gun"]))
    Mult = np.ones_like(W); Av = np.zeros(W.shape, bool)
    kol = []
    for s in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        jt = PT["syms"].index(s); jf = P["syms"].index(s); kol.append(jt)
        pv = np.full(len(gunT), np.nan); pv[vd] = pct[di[vd], jf]
        Av[:, jt] = np.isfinite(pv) & (W[:, jt] > 0)
        Mult[:, jt] = np.where(np.isfinite(pv) & (pv > ust), 0.5, 1.0)

    def kos(Wx):
        eq, g = H.sim_portfoy(PT["O"][b:], PT["C"][b:], Wx, maliyet=0.0007)
        gg = gunT[:len(g)]; r = {}
        for bl in bolumler:
            r[bl] = H.portfoy_rapor(eq, g, gg, bl)
            aa, bb = H.BOLUM[bl]; ms = (gg >= aa) & (gg < bb) & (gg >= _ms("2019-12-15"))
            r[bl + "_fundingdonemi"] = H.portfoy_rapor(eq, g[ms], gg[ms])
        return r
    taban = kos(W); ov = kos(W * Mult)
    etkin = {}
    for bl in bolumler:
        aa, bb = H.BOLUM[bl]; ms = (gunT >= aa) & (gunT < bb)
        etkin[bl] = {PT["syms"][j]: int(((Mult[:, j] < 1) & Av[:, j] & ms).sum()) for j in kol}
    rng = np.random.default_rng(1); pl = {k: [] for k in taban}
    for sd in range(seeds):
        Mp = np.ones_like(W)
        for bl in bolumler:
            aa, bb = H.BOLUM[bl]; ms = (gunT >= aa) & (gunT < bb)
            for j in kol:
                cand = np.flatnonzero(Av[:, j] & ms); k = etkin[bl][PT["syms"][j]]
                if k and len(cand):
                    Mp[rng.choice(cand, min(k, len(cand)), replace=False), j] = 0.5
        rp = kos(W * Mp)
        for k in rp:
            pl[k].append(rp[k])
    plo = {k: dict(sharpe_ort=round(float(np.mean([x["sharpe"] for x in v])), 3),
                   sharpe_p05_p95=[round(float(np.percentile([x["sharpe"] for x in v], q)), 3) for q in (5, 95)],
                   cagr_ort=round(float(np.mean([x["cagr"] for x in v])), 4),
                   maxdd_ort=round(float(np.mean([x["maxdd"] for x in v])), 3)) for k, v in pl.items()}
    out = dict(taban=taban, overlay=ov, plasebo=plo, yarilanan_pozisyonlu_gun=etkin, pencere=win, ust=ust)
    if kaydet:
        for bl in bolumler:
            H.kaydet(AILE, f"H2_overlay_s3bnb_vt30_{win}_p{int(ust*100)}", dict(pencere=win, ust=ust, carpan=0.5, taban="tsmom2 s3_orij_bnb hedef=0.3"),
                     bl, dict(overlay=ov[bl], taban=taban[bl], overlay_fd=ov[bl + "_fundingdonemi"],
                              taban_fd=taban[bl + "_fundingdonemi"], plasebo=plo[bl], plasebo_fd=plo[bl + "_fundingdonemi"]))
            H.kaydet(AILE, f"PLASEBO_H2_overlay_{win}_p{int(ust*100)}", dict(seeds=seeds, pencere=win), bl, plo[bl])
    for k in taban:
        print(f"OVERLAY {k}: taban S={taban[k].get('sharpe')} cagr={taban[k].get('cagr')} dd={taban[k].get('maxdd')} | "
              f"overlay S={ov[k].get('sharpe')} cagr={ov[k].get('cagr')} dd={ov[k].get('maxdd')} | plasebo {plo[k]}")
    print("yarilanan gun:", etkin)
    return out


# =================================================================== RAPOR
ADAYLAR = ["H1a_hepacik__genis", "H1b_esik10__genis", "H1c_top10__genis"]
KOMSULAR = {
    "H1a_hepacik__genis": "cekirdek-13 evren train S=4.74 cagr=5.9% dd=4.0% (ayni kural, gecti). Tek parametresiz kural.",
    "H1b_esik10__genis": "E=5/10/20/30 genis: S 7.52/7.45/7.01/6.00, cagr 7.5/7.4/6.5/5.3%; cekirdek 7.20/7.15/6.64/5.58 "
                         "(E30 cekirdek cagr 4.5% -> tek kalan). 'yeniden dagit (cap 0.10)' varyanti S 6.2-7.1, "
                         "maliyet 1.4-3.1%/yil daha yuksek. Genis parametre bolgesi pozitif.",
    "H1c_top10__genis": "K=3/5/10 genis S 3.94/4.58/5.80, cekirdek 3.93/5.47/4.77. Rastgele-K plasebo (20 seed) genis: "
                        "S ort -0.87/-0.46/0.88 (max -0.43/-0.05/1.29) -> secim plaseboyu her K'da geciyor; plasebo negatif "
                        "cunku haftalik rotasyon maliyeti (%3-6/yil) funding'i yiyor.",
}
DUSENLER = [
    "VALID 2025: uc adayin UCU de aday olcutunu gecemedi. H1a genis cagr -1.2% (S -3.45), H1b esik10 +0.2% (S 1.71 ama "
    "ort notional 0.10, cagr<5%), H1c top10 -1.0% (S -2.04; funding +2.4%/yil, maliyet -3.3%/yil).",
    "H2 funding yuzdeligi -> ileri spot getiri: 12 testin (3g/7g x 1/3/7g x yuksek/dusuk) HICBIRI Bonferroni ile anlamli degil "
    "(ay-blok bootstrap GA'si hepsinde 0'i iceriyor). Zaman sabit etkisiyle farklar -0.3%..+0.03% (~0).",
    "H2 ham yon hipotezin TERSI: yuksek funding (>%90) sonrasi 7g getiri orta bolgeden +2.6% FAZLA (ters donus degil); "
    "dusuk funding sonrasi da +1.3% fazla -> U-sekli, piyasa rejimi etkisi, coin-ozgu sinyal yok.",
    "H2 overlay (tsmom2 s3_orij_bnb vt30, yuzdelik>%90 iken agirlik x0.5): train S 1.58 -> 1.51 (7g) / 1.52 (3g), "
    "cagr 35.7% -> 30.0%/30.7%, maxDD 27.5% -> 26.1%/26.5%. Rastgele ayni sayida gun yarilama plasebosu S ort 1.51/1.53 "
    "-> overlay = sadece maruziyet azaltma, bilgi yok. VALID'e goturulmedi.",
    "H1a genis 'hep acik' 2022'de -4.2%, 2025'te -1.2%: yeni listelenen/kalabalik-short altcoinlerin negatif funding'i "
    "(2025: AVNT -1.0%, ENA -0.3%, WAL -0.2% sermaye/yil) BTC/ETH/LTC/LINK katkisini (her biri ~+0.1%) siliyor.",
]
OGRENILEN = [
    "Carry'nin getirisi REJIME bagli ve yillar icinde eriyor: H1a genis yillik (sermaye, notional=0.5) 2020 +10.8%, 2021 +20.4%, "
    "2022 -4.2%, 2023 +1.9%, 2024 +5.8%, 2025 -1.2%. TRAIN ortalamasi (6.4%) 2020-21 boga funding'inden geliyor. "
    "BTC 7g ort funding yillik: 2021 30.6% -> 2025 5.1% (notional); sermayeye ~2.5%.",
    "TRAIN'deki Sharpe 4-7 yaniltici: vol %0.9-1.4/yil cok dusuk oldugu icin kucuk ama istikrarli getiri Sharpe'i sisiriyor. "
    "Karar icin mutlak getiriye bakilmali; o da 2023-2025'te nakit faizinin (USDT/T-bill ~%4-5) altinda.",
    "Baz kuyrugu gercek ama kucuk (kaldiracsiz): 2022-11-10 FTX gunu H1a genis -1.09% (SOL spot-perp getiri farki -25.3% "
    "notional -> -0.60% sermaye, + funding -0.41%). 2021-05 cokusu -0.22%, 2022-06 3AC -0.35%. Esikli kural (H1b) bu gunlerin "
    "ucunde de ~0 (negatif funding'de pozisyon yok). En buyuk kayip kaynagi baz degil NEGATIF FUNDING: H1a genis train'de "
    "funding<0 olan 479 gunde toplam -7.7%.",
    "Esik (H1b) mantikli bir filtre: negatif-funding kaybini 479 gun/-7.7%'den 68 gun/-0.75%'e indiriyor ve DD'yi %4.4 -> %0.5 "
    "yapiyor; ama 2025'te esigin uzerinde cok az coin kaldigi icin sermayenin %90'i bos durdu (getiri +0.2%).",
    "Funding yuzdeligi coin-ozgu yonlu sinyal tasimiyor (zaman sabit etkisi ~0); ham 'yuksek funding sonrasi yuksek getiri' "
    "sadece boga piyasasi gunlerini isaretliyor. Trend portfoyunda maruziyet kismak plasebo kadar zarar veriyor.",
]
KARAR = dict(
    kagit_testnet_eklensin_mi="HAYIR (getiri stratejisi olarak). 'Cok kazandiran' bir sonuc yok: VALID 2025'te en iyi aday "
        "+0.2%/yil, digerleri negatif. En fazla H1b_esik10 (7g ort yillik funding > %10 ac, < %5 kapa, t+1) DUSUK ONCELIKLI "
        "bir 'funding rejim izleyici' olarak kagit-takibe konabilir: sadece funding rejimi 2021 gibi yukselirse devreye girer.",
    nasil_uygulanir="Coin basina: spot al (USDT-M perp ile ayni notional) + USDT-M perp short 1x. Sermaye yari spot yari perp "
        "teminati. Karar gunluk 00:00 UTC kapanis sonrasi, islem hemen ardindan (t+1 acilis). Esik: son 21 funding odemesi (7g) "
        "ort * 3 * 365 > E. Pozisyon basi notional = 0.5 / evren_boyu. Modelde OLMAYAN riskler: (1) fiyat ~%100 yukselirse 1x "
        "short teminati biter -> spot'tan perp cuzdanina periyodik transfer/yeniden dengeleme sart; (2) tek borsa karsi taraf riski "
        "(tum sermaye Binance'te); (3) funding araligi 8h->4h/1h degisimleri; (4) delist (FTT perp 2022-11'de dondu - veri filtresi "
        "gerekti). Teminat/USDT faizi sifir varsayildi.",
    testnet="Binance SPOT testnet (testnet.binance.vision) ve USDT-M FUTURES testnet (testnet.binancefuture.com) AYRI sistemler: "
        "ayri hesap/API anahtari, aralarinda transfer yok, testnet fiyat/funding ve derinlik mainnet'i yansitmiyor. Yani delta-notr "
        "carry PnL'i testnet'te ANLAMSIZ; testnet sadece emir mekanigini (iki bacagin eszamanli acilip kapanmasi) sinamaya yarar. "
        "Dogru kagit takibi: mainnet PUBLIC funding + spot/perp fiyatlariyla gunluk hipotetik PnL defteri (anahtarsiz).",
    beklenen_getiri_araligi="Sermaye bazinda (notional=0.5, kaldiracsiz, maliyet sonrasi): mevcut rejimde (2023-2025) yillik "
        "yaklasik -1% .. +6%, merkez ~+1-2% (H1a cekirdek tanimlayici 2025: +1.1%). Yuksek-funding boga rejiminde (2020-21 benzeri) "
        "+10..+20%. Vol ~%1/yil, maxDD tipik <%5 (baz kuyrugu tek gunde ~-1%). Nakit faizi (~%4) bu araligin ortasinin USTUNDE -> "
        "firsat maliyeti hesaba katilinca beklenen fazla getiri ~0 veya negatif.",
)
ONERI = ("Funding ailesi getiri kaynagi olarak KAPANABILIR: carry 2023+ rejiminde nakit faizini gecmiyor, yonlu funding sinyali "
         "(H2) coin-ozgu bilgi tasimiyor. Bir sonraki ajan icin: (1) carry'yi ancak 'nakit faizi uzerine eklenen' bir katman "
         "olarak dusun - ornegin teminati faizli varlikta tutma (Binance portfolio margin/earn) ve bu faizin getiriye eklenmesi; "
         "(2) H1b esik kuralini HOLDOUT'a goturmek yerine rejim izleyici olarak kagitta birak; (3) H2 icin tekrar deneme yapma - "
         "zaman sabit etkisi ~0; (4) tsmom trend portfoyu uzerine funding overlay'i denemek anlamsiz (plasebo ile ayni).")
NOTLAR = [
    "Sermaye varsayimi: sermayenin yarisi spot alimi, yarisi 1x perp teminati -> notional = 0.5 sermaye; tum getiriler SERMAYE "
    "bazinda (notional getirisinin yarisi). Kaldirac yok. Maliyet 0.19% notional tek yon (spot taker 0.10 + perp taker 0.05 + "
    "kayma 0.02 x 2 bacak). Teminat/USDT faizi yok. Rebalans transfer maliyeti yok.",
    "Simulasyon: harness.sim_portfoy, her coin icin sentetik carry endeksi I (gunluk getiri = gun funding toplami + spot acilis "
    "getirisi - perp acilis getirisi). Funding odemesi (gun_d 00:00, gun_d+1 00:00] araligindaysa gun_d'ye yazilir. W t kapanisinda "
    "bilinir, t+1 acilisinda uygulanir.",
    "Evren 'genis': 13 cekirdek U her ay basi bir onceki gunun 90g ort spot dolar hacmine gore ilk-20 (sadece o gun funding + perp "
    "verisi olanlar). Ort 21.1 coin. Aday havuzu: 2019-06..2025-12 aylik spot hacim ilk-25 birlesimi (153 sembol, hepsinin "
    "funding verisi var). data1d_all bugunun listesi -> LUNA/SHIB gibi eski buyukler yok (hayatta kalma yanliligi).",
    "Veri temizligi: delist sonrasi donmus perp mumlari (hacim 0 veya h==l; FTT 2023+ fiyat 1.59 sabit) ve |perp/spot-1|>%30 "
    "gunler gecersiz sayildi. Gercek FTX gunu (SOL 2022-11-10 baz -25%) KORUNDU.",
    "HOLDOUT: indirilen 2026 funding/mum verisi panel kurulurken ts<2026-01-01 kesildi; harness spot verisi zaten kesik. "
    "LAB_HOLDOUT ayarlanmadi.",
    "Defter seffafligi: ilk H1 kosusu bir hata (NaN yayan rolling mean -> bos evren, tum sonuclar 0) nedeniyle 26 anlamsiz sifir "
    "kayit yazdi; bunlar defterden cikarildi (yedek: scratchpad/defter_yedek_funding_bug.jsonl). Ayrica VALID sonrasi "
    "tanimlayici incelemede H1a_hepacik__cekirdek VALID hesaplandi (+1.1%) ve 'TANIMLAYICI' notuyla kaydedildi - aday secimini "
    "etkilemedi, 3 aday sinirinin disinda.",
    "H2 istatistik: ileri getiri t+1 acilis -> t+1+h acilis; %0.5/%99.5 winsorize; fark = grup ort - orta ort (dummy panel "
    "regresyonu ile ozdes); GA iki bootstrap ile (coin yeniden ornekleme ve ay-blok), Bonferroni alfa=0.05/12; anlamli = iki GA "
    "da 0'i dislarsa. Zaman sabit etkili (kesitsel demean) versiyon ayrica raporlandi.",
]


def faz_yaz():
    S = H.SCRATCH
    h1 = json.load(open(os.path.join(S, "_funding_h1_train.json")))
    h2 = json.load(open(os.path.join(S, "_funding_h2_train.json")))
    ov = json.load(open(os.path.join(S, "_funding_overlay_train.json")))
    va = json.load(open(os.path.join(S, "_funding_valid.json")))
    d = [k for k in H.defter_oku() if k.get("aile") == AILE]
    en3 = []
    for ad in ADAYLAR:
        r = h1["h1"][ad]
        en3.append(dict(ad=ad, params=r["params"], train=r["train"], valid=va["h1"][ad]["valid"],
                        aday_olcutu=dict(train=aday_mi(r["train"]), valid=aday_mi(va["h1"][ad]["valid"])),
                        kuyruk_olaylari_train=r["olaylar"], komsular=KOMSULAR[ad]))
    tablo = {ad: dict(cagr=r["train"]["cagr"], vol=r["train"]["vol"], sharpe=r["train"]["sharpe"], maxdd=r["train"]["maxdd"],
                      en_kotu_ay=r["train"]["en_kotu_ay"], katki=r["train"]["katki_yillik"], notional=r["train"]["ort_notional"],
                      yillar=r["train"]["yillar"], funding_negatif_gunler=r["train"]["funding_negatif_gunler"],
                      btc_korelasyon=r["train"]["btc_korelasyon"], aday=aday_mi(r["train"])) for ad, r in h1["h1"].items()}
    out = dict(
        aile=AILE, denenen=len({(k["ad"], k["bolum"]) for k in d}), defter_kayit=len(d), kaynak="1d (funding 8h->gunluk)",
        bolum="TRAIN 2019-09..2024-12 | VALID 2025 (tek sefer, 3 aday) | HOLDOUT 2026 kapali",
        aday_olcutu="TRAIN ve VALID'de Sharpe>1.0 VE yillik getiri>%5 VE maxDD<%15 (sermaye bazinda)",
        en_iyi_3=en3,
        h1_train_tablo=tablo,
        h1_plasebo=dict(train=h1["plasebo"], valid=va.get("plasebo")),
        kiyas=dict(nakit=0.0, btc_altut=h1["btc"]),
        h2_test_train=h2["train"],
        h2_overlay_train=ov,
        coin_funding_train=h1["coin"], coin_funding_valid=va["coin"],
        dusenler=DUSENLER, ogrenilen=OGRENILEN, karar=KARAR, oneri=ONERI, notlar=NOTLAR,
    )
    p = os.path.join(LAB, "sonuc_funding.json")
    json.dump(_js(out), open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("yazildi:", p, "| denenen:", out["denenen"], "| defter:", out["defter_kayit"])


def _js(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="h1", choices=["indir", "h1", "h2", "valid", "yaz"])
    ap.add_argument("--adaylar", default="")
    a = ap.parse_args()
    if a.faz == "indir":
        faz_indir()
    elif a.faz == "h1":
        P = panel(); U = evren("genis")
        print(f"panel {len(P['syms'])} sembol, {len(P['gun'])} gun; genis evren ort {U.sum(1)[U.sum(1)>0].mean():.1f} coin")
        res = dict(h1=faz_h1(("train",)), btc=btc_altut(("train", "valid")), coin=coin_ozet("train"),
                   top20=P["top20_tarihce"])
        res["plasebo"] = {f"top{k}_{ev}": plasebo_h1c(ev, k) for ev in ("genis", "cekirdek") for k in (3, 5, 10)}
        json.dump(_js(res), open(os.path.join(H.SCRATCH, "_funding_h1_train.json"), "w"), indent=1)
    elif a.faz == "h2":
        res = dict(train=h2_test("train"))
        json.dump(_js(res), open(os.path.join(H.SCRATCH, "_funding_h2_train.json"), "w"), indent=1)
    elif a.faz == "valid":
        # TEK SEFER, en fazla 3 aday (TRAIN'de secilmis). Plasebo: H1c adayi varsa ayni K ile.
        ad = [x for x in a.adaylar.split(",") if x][:3]
        res = dict(h1=faz_h1(("valid",), sadece=ad), coin=coin_ozet("valid"))
        res["plasebo"] = {}
        for x in ad:
            if x.startswith("H1c_top"):
                k = int(x.split("_top")[1].split("__")[0]); ev = x.split("__")[1]
                res["plasebo"][x] = plasebo_h1c(ev, k, ("valid",))
        json.dump(_js(res), open(os.path.join(H.SCRATCH, "_funding_valid.json"), "w"), indent=1)
    elif a.faz == "yaz":
        faz_yaz()


if __name__ == "__main__":
    main()
