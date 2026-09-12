"""tsmom - gunluk zaman-serisi momentum toplulugu (PORTFOY bazli, kaynak=1d).

Hipotezler (ana ajan gorevi):
  1) Coklu-MA oylamasi (MA20/50/100/200 + getiri 30/60/90 isareti), agirlik = pozitif oy orani
  2) Volatilite hedefleme (hedef_vol / gerceklesen_vol30, ust sinir 1.5x)
  3) Evren: sabit BTC/ETH/BNB  vs  90g dolar hacmine gore ilk 5/10/20
  4) Long/short (ayi rejiminde short, funding) vs nakit
  5) Cikis histerezisi (MA'nin %2-3 altina inince cik) vs ham kesisim

Kurallar: sadece harness.py; sinyal t kapanisinda -> islem t+1 acilista (sim_portfoy semantigi);
gostergeler yalniz gecmise bakar; maliyet harness varsayilani (0.0007/birim ciro).

Kullanim:
  PYTHONIOENCODING=utf-8 python tsmom.py --faz train
  PYTHONIOENCODING=utf-8 python tsmom.py --faz valid      # sadece ADAYLAR (<=3) + kiyaslar
  PYTHONIOENCODING=utf-8 python tsmom.py --faz rapor      # sonuc_tsmom.json yaz
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "tsmom"
CEKIRDEK = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
# Ortak analiz baslangici: MA200 + 90g hacim penceresi isindiktan sonra (veri 2017-08-17'de basliyor)
BAS_MS = int(datetime(2018, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)
VOL_TABAN = 0.10          # yillik vol alt siniri (bolme patlamasini onler)
VOL_UST = 1.5             # vol hedefleme ust sinir carpani


# ----------------------------------------------------------------- yardimci (hepsi gecmise bakar)
def ffill(A):
    M = ~np.isnan(A)
    idx = np.where(M, np.arange(A.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.take_along_axis(A, idx, axis=0)
    # listelenmeden onceki bolum nan kalsin
    ilk = np.where(M.any(0), M.argmax(0), A.shape[0])
    out[np.arange(A.shape[0])[:, None] < ilk[None, :]] = np.nan
    return out


def roll_mean(A, N):
    n, m = A.shape
    X = np.nan_to_num(A, nan=0.0); M = (~np.isnan(A)).astype(float)
    cs = np.cumsum(np.vstack([np.zeros((1, m)), X]), 0)
    cm = np.cumsum(np.vstack([np.zeros((1, m)), M]), 0)
    s = cs[N:] - cs[:-N]; c = cm[N:] - cm[:-N]
    out = np.full((n, m), np.nan)
    out[N - 1:] = np.where(c >= N, s / N, np.nan)
    return out


def roll_std(A, N):
    m1 = roll_mean(A, N); m2 = roll_mean(A * A, N)
    v = np.maximum(m2 - m1 * m1, 0.0)
    return np.sqrt(v * N / max(N - 1, 1))


def hyst(C, MA, band):
    """C>MA ile gir, C<MA*(1-band) olunca cik; arada onceki durumu koru (whipsaw azaltma)."""
    n, m = C.shape
    ev = np.where(C > MA, 1, np.where(C < MA * (1 - band), -1, 0)).astype(np.int8)
    idx = np.where(ev != 0, np.arange(n)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    ok = idx >= 0
    st = np.take_along_axis(ev, np.where(ok, idx, 0), 0) == 1
    return st & ok


# ----------------------------------------------------------------- panel + gostergeler
_P = {}
def panel():
    if _P:
        return _P
    syms, gun, O, C, V = H.gunluk_panel(min_gun=200)
    Cf = ffill(C)
    R = np.full_like(Cf, np.nan); R[1:] = Cf[1:] / Cf[:-1] - 1
    dv = roll_mean(np.nan_to_num(V) * np.nan_to_num(Cf), 90)         # 90g ort dolar hacmi
    dv = np.where(np.isnan(Cf), np.nan, dv)
    _P.update(syms=syms, gun=gun, O=O, C=C, V=V, Cf=Cf, R=R, dv=dv, volc={},
              bas=int(np.searchsorted(gun, BAS_MS)))
    return _P


def volat(win=30):
    """yillik gerceklesen vol, t kapanisinda bilinir (gecmis win gun)"""
    P = panel()
    if win not in P["volc"]:
        P["volc"][win] = np.maximum(roll_std(P["R"], win) * np.sqrt(365), VOL_TABAN)
    return P["volc"][win]


_OY = {}
def oylar(ma_list, ret_list, band):
    k = (tuple(ma_list), tuple(ret_list), band)
    if k in _OY:
        return _OY[k]
    P = panel(); Cf = P["Cf"]
    vs = []
    for p in ma_list:
        ma = roll_mean(Cf, p)
        v = hyst(Cf, ma, band) if band > 0 else (Cf > ma)
        vs.append(np.where(np.isnan(ma) | np.isnan(Cf), np.nan, v.astype(float)))
    for p in ret_list:
        r = np.full_like(Cf, np.nan); r[p:] = Cf[p:] / Cf[:-p] - 1
        vs.append(np.where(np.isnan(r), np.nan, (r > 0).astype(float)))
    Vt = np.stack(vs)
    cnt = (~np.isnan(Vt)).sum(0)
    frac = np.where(cnt == len(vs), np.nansum(Vt, 0) / len(vs), np.nan)
    tam = cnt == len(vs)
    _OY[k] = (frac, tam)
    return frac, tam


def evren_maske(tam, cfg):
    """t gunu kapanisinda secilebilir semboller (yalniz gecmis veri)."""
    P = panel(); syms, Cf, dv = P["syms"], P["Cf"], P["dv"]
    vol = volat(cfg.get("volwin", 30))
    canli = tam & ~np.isnan(Cf) & ~np.isnan(vol)
    if cfg["evren"] == "sabit3":
        col = np.zeros(len(syms), bool)
        for s in (cfg.get("liste") or CEKIRDEK):
            if s in syms:
                col[syms.index(s)] = True
        return canli & col[None, :]
    N = cfg["top"]
    sk = np.where(canli & ~np.isnan(dv), dv, -np.inf)
    order = np.argsort(-sk, axis=1)[:, :N]
    m = np.zeros(sk.shape, bool)
    np.put_along_axis(m, order, True, axis=1)
    return m & canli & np.isfinite(sk)


def agirliklar(cfg):
    vol = volat(cfg.get("volwin", 30))
    frac, tam = oylar(cfg["ma"], cfg["ret"], cfg["band"])
    um = evren_maske(tam, cfg)
    s = np.nan_to_num(frac)
    if cfg["ls"]:
        s = 2 * s - 1
    if cfg["esik"] is not None:
        e = cfg["esik"]
        s = np.where(np.nan_to_num(frac) >= e, 1.0,
                     np.where(cfg["ls"] & (np.nan_to_num(frac) <= 1 - e), -1.0, 0.0))
    s = np.where(um, s, 0.0)
    K = um.sum(1)
    if cfg["taban"] == "invvol":
        iv = np.where(um, 1.0 / np.nan_to_num(vol, nan=1e9), 0.0)
        b = iv / np.maximum(iv.sum(1, keepdims=True), 1e-12)
    else:
        b = np.where(um, 1.0 / np.maximum(K, 1)[:, None], 0.0)
    W = b * s
    if cfg["hedef"]:
        W = W * np.clip(cfg["hedef"] / np.nan_to_num(vol, nan=1e9), 0, cfg.get("cap", VOL_UST))
    gross = np.abs(W).sum(1, keepdims=True)
    W = np.where(gross > 1.0, W / np.maximum(gross, 1e-12), W)
    if cfg.get("mind", 0) > 0:
        out = np.zeros_like(W); cur = np.zeros(W.shape[1])
        for t in range(len(W)):
            chg = np.abs(W[t] - cur) >= cfg["mind"]
            cur = np.where(chg, W[t], cur)
            out[t] = cur
        W = out
    return W


def kiyas_agirlik(cfg, tip):
    P = panel(); syms = P["syms"]
    if tip == "btc":
        W = np.zeros_like(P["Cf"]); j = syms.index("BTCUSDT")
        W[:, j] = np.where(np.isnan(P["Cf"][:, j]), 0.0, 1.0)
        return W
    frac, tam = oylar(cfg["ma"], cfg["ret"], cfg["band"])
    um = evren_maske(tam, cfg)
    K = np.maximum(um.sum(1), 1)[:, None]
    return np.where(um, 1.0 / K, 0.0)


# ----------------------------------------------------------------- kosu
def kos(cfg, bolumler=("train",), kaydet=True, ad=None):
    P = panel(); b = P["bas"]
    W = agirliklar(cfg)
    fund = cfg.get("funding", 0.0)
    eq, g = H.sim_portfoy(P["O"][b:], P["C"][b:], W[b:], funding=fund)
    gun = P["gun"][b:]
    out = {}
    for bl in bolumler:
        r = H.portfoy_rapor(eq, g, gun, bl)
        r["ciro"] = round(float(np.abs(np.diff(np.vstack([np.zeros((1, W.shape[1])), W[b:]]), axis=0)).sum(1).mean()), 4)
        r["ort_gross"] = round(float(np.abs(W[b:]).sum(1).mean()), 3)
        out[bl] = r
        if kaydet:
            H.kaydet(AILE, ad or cfg["ad"], cfg, bl, r)
    return out


def kiyaslar(cfg, bolumler, kaydet=True):
    P = panel(); b = P["bas"]; res = {}
    for tip in ("btc", "evren"):
        W = kiyas_agirlik(cfg, tip)
        eq, g = H.sim_portfoy(P["O"][b:], P["C"][b:], W[b:])
        for bl in bolumler:
            r = H.portfoy_rapor(eq, g, P["gun"][b:], bl)
            res[f"{tip}_{bl}"] = r
            if kaydet:
                H.kaydet(AILE, f"KIYAS_{tip}_{cfg.get('ad', cfg['evren'])}{cfg.get('top','')}", dict(kiyas=tip, **{k: v for k, v in cfg.items() if k != 'ad'}), bl, r)
    return res


def C(ad, **kw):
    d = dict(ad=ad, evren="sabit3", top=0, ma=[50], ret=[], band=0.0, esik=None,
             taban="esit", hedef=None, ls=False, mind=0.0, funding=0.0,
             volwin=30, cap=VOL_UST, liste=None)
    d.update(kw)
    return d


# ----------------------------------------------------------------- fazlar
def faz_train():
    cfgs = []
    # --- 0: temel (bilinen sonucun portfoy karsiligi)
    cfgs += [C("temel_ma50"), C("temel_ma20", ma=[20]), C("temel_ma100", ma=[100]), C("temel_ma200", ma=[200])]
    # --- 1: coklu-MA oylamasi
    cfgs += [C("oy_ma4", ma=[20, 50, 100, 200]),
             C("oy_ma4_ret3", ma=[20, 50, 100, 200], ret=[30, 60, 90]),
             C("oy_ret3", ma=[], ret=[30, 60, 90]) if False else C("oy_ma2_ret2", ma=[50, 200], ret=[60, 90]),
             C("oy_ma4_esik50", ma=[20, 50, 100, 200], esik=0.5),
             C("oy_ma4_esik75", ma=[20, 50, 100, 200], esik=0.75),
             C("oy_ma4_esik100", ma=[20, 50, 100, 200], esik=1.0),
             C("oy_ma4ret3_esik50", ma=[20, 50, 100, 200], ret=[30, 60, 90], esik=0.5),
             C("oy_ma4ret3_esik70", ma=[20, 50, 100, 200], ret=[30, 60, 90], esik=0.70)]
    # --- 5: histerezis
    for bd in (0.02, 0.03, 0.05):
        cfgs += [C(f"temel_ma50_hist{int(bd*100)}", band=bd),
                 C(f"oy_ma4_hist{int(bd*100)}", ma=[20, 50, 100, 200], band=bd)]
    # --- 2: volatilite hedefleme
    for hv in (0.20, 0.30, 0.40):
        cfgs += [C(f"ma50_vt{int(hv*100)}", hedef=hv),
                 C(f"oy_ma4_vt{int(hv*100)}", ma=[20, 50, 100, 200], hedef=hv)]
    cfgs += [C("ma50_vt30_mind05", hedef=0.30, mind=0.05),
             C("oy_ma4_vt30_mind05", ma=[20, 50, 100, 200], hedef=0.30, mind=0.05),
             C("ma50_invvol"), C("ma50_invvol_x", taban="invvol")]
    cfgs = [c for c in cfgs if c["ad"] != "ma50_invvol"]
    # --- 3: evren
    for N in (5, 10, 20):
        cfgs += [C(f"top{N}_ma50", evren="top", top=N),
                 C(f"top{N}_oy_ma4", evren="top", top=N, ma=[20, 50, 100, 200]),
                 C(f"top{N}_oy_ma4_vt30", evren="top", top=N, ma=[20, 50, 100, 200], hedef=0.30),
                 C(f"top{N}_ma50_invvol", evren="top", top=N, taban="invvol"),
                 C(f"top{N}_oy_ma4_hist3", evren="top", top=N, ma=[20, 50, 100, 200], band=0.03)]
    # --- 4: long/short
    for f in (0.0003, 0.0):
        tag = "f3" if f else "f0"
        cfgs += [C(f"ls_ma50_{tag}", ls=True, esik=0.5, funding=f),
                 C(f"ls_oy_ma4_{tag}", ma=[20, 50, 100, 200], ls=True, funding=f),
                 C(f"ls_top10_oy_ma4_{tag}", evren="top", top=10, ma=[20, 50, 100, 200], ls=True, funding=f)]
    return cfgs


B5 = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
# TRAIN sonrasi secilen 3 aday (VALID'e TEK sefer gider)
ADAYLAR = [
    C("A1_b3_ma50"),                                    # bilinen temel (kontrol)
    C("A2_b3_ma50_vt30", hedef=0.30),                   # en iyi risk-ayarli, komsulari genis pozitif
    C("A3_b5_ma50_vt30", liste=B5, hedef=0.30),         # daha genis sabit evren + vol hedefleme
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="train", choices=["train", "valid", "komsu", "rapor"])
    ap.add_argument("--filtre", default="")
    a = ap.parse_args()
    P = panel()
    print(f"panel: {len(P['syms'])} sembol, {len(P['gun'])} gun, analiz bas idx={P['bas']} "
          f"({datetime.fromtimestamp(P['gun'][P['bas']]/1e3, timezone.utc).date()})")

    if a.faz in ("train", "komsu"):
        cfgs = faz_train() if a.faz == "train" else faz_komsu()
        if a.filtre:
            cfgs = [c for c in cfgs if a.filtre in c["ad"]]
        # kiyaslar
        for e in ({"evren": "sabit3", "top": 0, "ma": [50], "ret": [], "band": 0.0},
                  {"evren": "top", "top": 5, "ma": [50], "ret": [], "band": 0.0},
                  {"evren": "top", "top": 10, "ma": [50], "ret": [], "band": 0.0},
                  {"evren": "top", "top": 20, "ma": [50], "ret": [], "band": 0.0}):
            k = kiyaslar(e, ("train",))
            print(f"KIYAS {e['evren']}{e['top']:>3}  btc={k['btc_train']}  evren={k['evren_train']}")
        rows = []
        for c in cfgs:
            r = kos(c, ("train",))["train"]
            rows.append((c["ad"], r))
            print(f"{c['ad']:<26} sharpe={r.get('sharpe'):>6} cagr={r.get('cagr'):>7} "
                  f"maxdd={r.get('maxdd'):>6} gun={r.get('gun')} ciro={r.get('ciro')} gross={r.get('ort_gross')}")
        print("\n--- train sharpe siralamasi ---")
        for ad, r in sorted(rows, key=lambda x: -(x[1].get("sharpe") or -9)):
            print(f"{ad:<26} {r.get('sharpe'):>6} dd={r.get('maxdd')}")

    elif a.faz == "rapor":
        yaz_rapor()

    elif a.faz == "valid":
        assert 0 < len(ADAYLAR) <= 3, "en fazla 3 aday"
        for e in (C("kiyas_b3"), C("kiyas_b5", liste=B5), C("kiyas_top10", evren="top", top=10)):
            k = kiyaslar(e, ("train", "valid"))
            print(f"KIYAS {e['ad']}\n   btc_tr={k['btc_train']}\n   btc_va={k['btc_valid']}\n"
                  f"   ev_tr={k['evren_train']}\n   ev_va={k['evren_valid']}")
        for c in ADAYLAR:
            r = kos(c, ("train", "valid"))
            print(f"\n== {c['ad']}\n  train={r['train']}\n  valid={r['valid']}")


def defter_al(ad, bolum):
    son = None
    for k in H.defter_oku():
        if k.get("aile") == AILE and k.get("ad") == ad and k.get("bolum") == bolum:
            son = k["rapor"]
    return son


def yaz_rapor():
    d = [k for k in H.defter_oku() if k.get("aile") == AILE]
    n_test = len({(k["ad"], k["bolum"]) for k in d if not k["ad"].startswith("KIYAS")})
    G = lambda ad, bl: defter_al(ad, bl)
    kiyas = {"btc_altut": {"train": G("KIYAS_btc_kiyas_b30", "train") or G("KIYAS_btc_sabit30", "train"),
                           "valid": G("KIYAS_btc_kiyas_b30", "valid")},
             "b3_esit_altut": {"train": G("KIYAS_evren_kiyas_b30", "train"), "valid": G("KIYAS_evren_kiyas_b30", "valid")},
             "b5_esit_altut": {"train": G("KIYAS_evren_kiyas_b50", "train"), "valid": G("KIYAS_evren_kiyas_b50", "valid")},
             "top10_esit_altut": {"train": G("KIYAS_evren_kiyas_top1010", "train"), "valid": G("KIYAS_evren_kiyas_top1010", "valid")}}
    komsu = {
        "A1_b3_ma50": "MA neighbors (train Sharpe): MA20 1.36 / MA50 1.49 / MA100 1.28 / MA200 1.20; "
                      "evren komsulari: BTC tek 1.27, BTC+ETH 1.21, ETH+BNB 1.41, b5 1.43, b6 1.29. "
                      "Genis bolge pozitif ama tepe b3'te -> b3 secimi kismen gecmise bakarak yapilmis (BNB katkisi buyuk).",
        "A2_b3_ma50_vt30": "hedef_vol 20/30/40 = 1.59/1.59/1.56; MA 30/40/50/60/70/80 @vt30 = 1.46/1.57/1.59/1.48/1.41/1.39; "
                           "vol penceresi 14/20/30/45/60/90 = 1.60/1.57/1.59/1.55/1.56/1.53; cap 1.0/1.5/2.0/3.0 farksiz; "
                           "ciro bandi (mind 0.05) 1.58. Komsu bolge tamamen 1.39-1.60 -> saglam plato.",
        "A3_b5_ma50_vt30": "hedef_vol 20/30/40 = 1.53/1.54/1.52; MA40/50/60 = 1.52/1.54/1.37. Train'de saglam, "
                           "VALID'de b3'un belirgin altinda (0.93 vs 1.39) -> evreni genisletmek bedelli.",
    }
    en_iyi = []
    for c in ADAYLAR:
        en_iyi.append(dict(ad=c["ad"], params={k: v for k, v in c.items() if k != "ad"},
                           train=G(c["ad"], "train"), valid=G(c["ad"], "valid"), komsular=komsu.get(c["ad"], "")))
    out = dict(
        aile=AILE, denenen=n_test, kaynak="1d", bolum="train<2025 | valid 2025",
        analiz_baslangici="2018-04-01 (MA200 + 90g hacim isinmasi sonrasi, tum konfigler ayni pencere)",
        kiyaslar=kiyas, en_iyi_3=en_iyi,
        hipotez_sonuclari={
            "H1_coklu_MA_oylamasi": "RED. MA20/50/100/200 (+getiri 30/60/90) oy orani agirligi train Sharpe 1.41-1.48; "
                                    "tek MA50 = 1.49. Oy esigi 0.75/1.0 DD'yi dusuruyor (0.41/0.29) ama Sharpe artmiyor. "
                                    "Oylama tek MA'ya gore bir sey eklemiyor.",
            "H2_volatilite_hedefleme": "KISMEN. hedef %20-40 + cap1.5: Sharpe 1.49->1.56-1.59, maxDD 0.50->0.19-0.36. "
                                       "ANCAK ort. gross 0.53 -> 0.18-0.35 dusuyor; ayni gross'a olceklendiginde "
                                       "(hedef %60-80) Sharpe 1.46-1.47 yani temelin ALTINDA. Yani DD kazanci buyuk olcude "
                                       "pozisyon kucultmeden geliyor; gercek sinyal kalitesi kazanci ~+0.10 Sharpe ve gurultu icinde. "
                                       "Yine de VALID'de birim gross basina getiri A2>A1 (1.00 vs 0.83) ve DD yarisi.",
            "H3_evren_genisletme": "RED, net. 90g dolar hacmine gore ilk 5/10/20 evren: train Sharpe 0.76-0.95 (vt30 ile 0.95-1.22), "
                                   "sabit 3 = 1.49. VALID'de b5 0.93 vs b3 1.39. Vol-agirlik (invvol) top-N'i biraz duzeltiyor "
                                   "ama sabit buyuk-cap sepetine yaklasmiyor. NOT: data1d_all bugun listede olan coinler -> "
                                   "top-N sonucu survivorship lehine yanli, yani gercekte daha da kotu.",
            "H4_long_short": "RED. Ayi rejiminde short: funding 0.0003/gun ile train Sharpe 0.77-0.78, funding=0 ile bile 0.91-0.95; "
                             "nakit (long-only) 1.43-1.49. Short bacagi 2020 ve 2023'te agir kaybediyor; ciro 2x. Funding olmasa bile kotu.",
            "H5_cikis_histerezisi": "RED (notr). Band %2/%3/%5: Sharpe 1.48/1.46/1.29 (temel 1.49); ciroyu 0.057->0.037/0.033/0.027 "
                                    "yani ~%40-50 dusuruyor. Maliyet cok daha yuksek olsaydi ise yarardi; mevcut 7bp'de kazanc yok.",
        },
        dusenler=[
            "coklu-MA/getiri oylamasi (8 varyant): tek MA50'yi gecemedi",
            "oy esigi 0.5/0.75/1.0: DD iyilesiyor, Sharpe iyilesmiyor",
            "evren top5/top10/top20 (hacim siralamasi, esit ve invvol): hepsi sabit-3'un cok altinda",
            "long/short (funding'li ve funding'siz, 6 varyant): long-only'nin cok altinda",
            "cikis histerezisi %2/%3/%5: Sharpe notr-negatif, sadece ciro dusuruyor",
            "vol hedeflemeyi ayni riske olceklemek (hedef %50-80): Sharpe 1.46-1.51, temelden iyi degil",
            "MA30/MA70/MA80, vol penceresi 90: kenarlarda bozulma basliyor ama hala pozitif",
        ],
        ogrenilen=[
            "Bu ailede tek gercek surucu 'buyuk-cap sabit sepet + tek trend filtresi'. Karmasiklik (oylama, "
            "histerezis, short, genis evren) her seferinde ya notr ya zararli.",
            "Evreni genisletmek tek yonlu kotu: top5->top10->top20 Sharpe 0.95->0.76->0.88 (train), b3->b5->b6 "
            "1.49->1.43->1.29. Altcoin trend takibi maliyet sonrasi ayakta kalmiyor.",
            "Vol hedefleme bir ALFA degil, bir RISK AYARI: ayni ortalama gross'a getirildiginde Sharpe artmiyor. "
            "Ama DD'yi yarilyor ve VALID'de birim risk basina getiriyi koruyor -> uygulanabilir bir kaldirac politikasi.",
            "Short bacagi bu evrende yapisal olarak zararli: kripto buyuk-cap'te ayi rejimi getirisi negatif degil, "
            "sadece volatil; funding sifir olsa bile Sharpe long-only'nin altinda.",
            "VALID 2025 kiyaslari cok zayif (BTC al-tut Sharpe 0.05, b5 al-tut 0.06, top10 0.17). Adaylarin 2025'te "
            "1.37/1.39/0.93 Sharpe uretmesi anlamli; ama tek yil ve n_gun=363, tek gozlem gibi davranilmali.",
            f"Defterde bu ailede {n_test} konfigurasyon var; coklu-test riski yuksek, 0.05-0.10 Sharpe farklari gurultudur.",
        ],
        oneri="A1 (b3 MA50 esit agirlik) ve A2 (ayni sinyal + hedef vol %30, cap 1.5) HOLDOUT 2026'ya goturulmeye deger; "
              "ikisi de train+valid'de her iki kiyasi hem Sharpe hem maxDD'de geciyor ve komsu parametre bolgesi genis pozitif. "
              "A2'yi tercih et: ayni Sharpe, yari maxDD, %33 daha az ciro. Bir sonraki ajan icin: (a) evren genisletme "
              "yonunu BIRAK - bes ayri sekilde denendi, hepsi kotu; (b) bunun yerine sinyalin uygulanmasina bak: "
              "haftalik/2-gunluk rebalans (ciro 7bp'de dusuk ama gunluk gurultu var), kismi cikis, veya BTC rejimini "
              "(harness.rejim_at) tum sepete kapi olarak uygulamak; (c) hedef vol'u zaman-degisken degil sabit tutmanin "
              "maliyeti test edilmeli; (d) b3 sepetinin kendisi gecmise bakarak secilmis - 2018'de bilinebilir bir kural "
              "(orn. o gunun ilk-3 piyasa degeri) ile ayni sepetin cikip cikmadigi dogrulanmali.",
        notlar=[
            "sim_portfoy funding'i gross'a uygular; long/short varyantinda short-only funding ayristirilamadigi icin "
            "funding hem long hem short bacaga yazildi (kotumser). Kontrol icin funding=0 varyanti da kosuldu, sonuc degismedi.",
            "Tum konfigler ayni pencerede (2018-04-01 sonrasi) kosuldu; maliyet harness varsayilani 0.0007/birim ciro.",
            "top-N evreni bugunku vadeli listesinden turedigi icin survivorship lehine yanli; gercek performans daha kotu.",
        ],
    )
    p = os.path.join(LAB, "sonuc_tsmom.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("yazildi:", p, "| denenen:", n_test)
    return out


def faz_komsu():
    """Kazanan bolgenin (MA50 + vol hedefleme, sabit 3 coin) komsulari + esit-risk kiyasi."""
    cfgs = []
    # esit-risk: vol hedefini yukselterek ort_gross'u temel_ma50 (0.53) seviyesine cikar
    for hv in (0.50, 0.60, 0.70, 0.80):
        cfgs.append(C(f"ma50_vt{int(hv*100)}", hedef=hv))
    # MA komsulari (vt30 ve vt60'ta)
    for p in (30, 40, 60, 70, 80):
        cfgs += [C(f"ma{p}_vt30", ma=[p], hedef=0.30), C(f"ma{p}_vt60", ma=[p], hedef=0.60)]
    # vol penceresi komsulari
    for w in (14, 20, 45, 60, 90):
        cfgs += [C(f"ma50_vt30_w{w}", hedef=0.30, volwin=w), C(f"ma50_vt60_w{w}", hedef=0.60, volwin=w)]
    # ust sinir komsulari
    for cp in (1.0, 2.0, 3.0):
        cfgs.append(C(f"ma50_vt60_cap{cp}", hedef=0.60, cap=cp))
    # histerezis + vol hedefleme, ciro bandi
    cfgs += [C("ma50_vt60_hist2", hedef=0.60, band=0.02), C("ma50_vt60_hist3", hedef=0.60, band=0.03),
             C("ma50_vt60_mind05", hedef=0.60, mind=0.05), C("ma50_vt60_mind10", hedef=0.60, mind=0.10),
             C("oy_ma4_vt60", ma=[20, 50, 100, 200], hedef=0.60),
             C("oy_ma4ret3_vt60", ma=[20, 50, 100, 200], ret=[30, 60, 90], hedef=0.60)]
    # evren saglamligi (sabit liste degisimi)
    listeler = dict(btc=["BTCUSDT"], btceth=["BTCUSDT", "ETHUSDT"],
                    b3=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                    b5=["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"],
                    b6=["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT"],
                    ethbnb=["ETHUSDT", "BNBUSDT"])
    for k, L in listeler.items():
        cfgs += [C(f"{k}_ma50_vt60", liste=L, hedef=0.60), C(f"{k}_ma50", liste=L)]
    # top-N + vol hedefleme yuksek hedefle (evren hipotezini adil test et)
    for N in (5, 10, 20):
        cfgs.append(C(f"top{N}_ma50_vt60", evren="top", top=N, hedef=0.60))
    return cfgs


if __name__ == "__main__":
    main()
