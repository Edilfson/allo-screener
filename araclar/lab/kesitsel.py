"""AILE: kesitsel  -- gunluk kesitsel momentum, portfoy bazli.

Hipotezler (Liu & Tsyvinski 2021; Jegadeesh-Titman 1993):
  H1  K gunluk getiri (K in {14,30,60,90}), son `skip`=7 gunu atla (kisa vadeli tersine donus),
      en yuksek ilk L coin esit agirlik LONG, haftalik yeniden dengeleme.
  H2  Long-short: en iyi L LONG, en kotu L SHORT (vadeli, funding=0.0003/gun).
  H3  BTC rejim filtresi: BTC kapanis > MA50 iken pozisyon, degilse nakit.
  H4  Volatilite hedefleme: agirlik ~ 1/vol30, toplam |w| <= 1.

Evren: her gun t'de, son 30 gunun ORTALAMA USDT hacmi (panel V kolonu zaten quote/USDT hacmi)
       ile ilk U coin. Evren GECMIS hacme gore secilir -> hayatta-kalma yanliligi azalir.
       Ek sart: o gune kadar coin gecmisi >= 120 gun ve o gun fiyat mevcut.

Ileriye bakma yok: t gununun KAPANISINDA bilinen bilgiyle W[t] kurulur; harness.sim_portfoy
W[t]'yi t+1 acilisinda uygular ve getiriyi O[t+1]->O[t+2] arasindan alir.

Bolumler kaynak="1d": train < 2025-01-01, valid 2025.
Kiyaslar: ayni evrenin esit-agirlik al-tutu (ew<U>) ve BTC al-tut (btc) -- ayni sim_portfoy,
ayni maliyet (0.0007 tek yon).

Kullanim:
  python kesitsel.py --faz train      # tam grid, sadece TRAIN raporu, hepsi deftere
  python kesitsel.py --faz valid      # SADECE FINAL adaylar + kiyaslar, TEK sefer
  python kesitsel.py --faz kiyas      # kiyaslarin train+valid raporu
  python kesitsel.py --tek H1_K30_U50_L10 --faz train
"""
import argparse, json, os, sys
import numpy as np
import harness as H

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "kesitsel"
SKIP = 7          # son 7 gun atlanir (kisa vadeli tersine donus)
MIN_HIST = 120    # coin gecmisi >= 120 gun
ADV_WIN = 30      # evren icin hacim penceresi
VOL_WIN = 30      # volatilite penceresi
MALIYET = 0.0007  # sim_portfoy varsayilani -- DEGISTIRILMEZ
FUNDING_LS = 0.0003

_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = H.gunluk_panel(min_gun=MIN_HIST)
    return _PANEL


# ------------------------------------------------------------------ yardimcilar
def _roll_mean(X, w):
    """nan'lari 0 sayarak gecmise bakan hareketli ortalama; ilk w-1 satir nan."""
    Z = np.nan_to_num(X, nan=0.0)
    cs = np.cumsum(Z, axis=0)
    out = np.full_like(Z, np.nan, dtype=float)
    out[w - 1:] = (cs[w - 1:] - np.vstack([np.zeros((1, Z.shape[1])), cs[:-w]])) / w
    return out


def onhesap():
    """Panelden turetilen, sadece GECMISE bakan matrisler."""
    syms, gun, O, C, V = panel()
    n, m = C.shape
    gecerli = ~np.isnan(C)
    hist = np.cumsum(gecerli, axis=0)                 # hist[t,j]: t dahil gecerli gun sayisi
    adv = _roll_mean(np.where(gecerli, V, np.nan), ADV_WIN)   # 30g ort USDT hacmi (t dahil)
    # gunluk getiri ve 30g volatilite (t dahil, gecmise bakar)
    Cp = np.vstack([np.full((1, m), np.nan), C[:-1]])
    r1 = C / Cp - 1.0
    R = np.nan_to_num(r1, nan=0.0)
    m1 = _roll_mean(R, VOL_WIN)
    m2 = _roll_mean(R ** 2, VOL_WIN)
    vol = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    # BTC MA50 rejimi: t kapanisinda bilinir
    jb = syms.index("BTCUSDT")
    cb = C[:, jb]
    mab = _roll_mean(cb.reshape(-1, 1), 50).ravel()
    rejim = np.where(cb > mab, 1, 0)
    rejim[np.isnan(mab) | np.isnan(cb)] = 0
    return dict(syms=syms, gun=gun, O=O, C=C, V=V, gecerli=gecerli, hist=hist,
                adv=adv, vol=vol, rejim=rejim, jb=jb, n=n, m=m)


def _agirlik(idx, vol_t, voltgt, butce):
    """Secilen idx'lere butce (toplam |w|) dagit."""
    if len(idx) == 0:
        return {}
    if voltgt:
        iv = 1.0 / np.maximum(vol_t[idx], 1e-4)
        iv = np.where(np.isfinite(iv), iv, 0.0)
        if iv.sum() <= 0:
            iv = np.ones(len(idx))
        w = butce * iv / iv.sum()
    else:
        w = np.full(len(idx), butce / len(idx))
    return dict(zip(idx, w))


def build_W(P, K=30, U=50, L=10, rebal=7, mod="long", rejim=False, voltgt=False, skip=SKIP):
    """t kapanisinda bilinen bilgiyle agirlik matrisi. mod: 'long' | 'ls' | 'ew'."""
    n, m = P["n"], P["m"]
    C, adv, hist, gecerli, vol = P["C"], P["adv"], P["hist"], P["gecerli"], P["vol"]
    W = np.zeros((n, m))
    t0 = max(ADV_WIN, skip + K + 1, VOL_WIN, 50) + 1
    baz = np.zeros(m)
    for t in range(t0, n):
        if (t - t0) % rebal == 0:
            elig = gecerli[t] & (hist[t] >= MIN_HIST) & np.isfinite(adv[t]) & (adv[t] > 0)
            ei = np.flatnonzero(elig)
            if len(ei) == 0:
                baz = np.zeros(m)
                continue
            # evren: gecmis 30g hacme gore ilk U
            ev = ei[np.argsort(-adv[t, ei])[:U]]
            baz = np.zeros(m)
            if mod == "ew":
                for j, w in _agirlik(ev, vol[t], voltgt, 1.0).items():
                    baz[j] = w
            else:
                mom = C[t - skip, ev] / C[t - skip - K, ev] - 1.0
                ok = np.isfinite(mom)
                ev2, mom = ev[ok], mom[ok]
                gerek = 2 * L if mod == "ls" else L
                if len(ev2) < gerek:
                    baz = np.zeros(m)
                    continue
                srt = np.argsort(-mom)
                if mod == "long":
                    for j, w in _agirlik(ev2[srt[:L]], vol[t], voltgt, 1.0).items():
                        baz[j] = w
                else:  # ls
                    for j, w in _agirlik(ev2[srt[:L]], vol[t], voltgt, 0.5).items():
                        baz[j] = w
                    for j, w in _agirlik(ev2[srt[-L:]], vol[t], voltgt, 0.5).items():
                        baz[j] = -w
        W[t] = baz * (P["rejim"][t] if rejim else 1)
    return W


def build_W_btc(P):
    W = np.zeros((P["n"], P["m"]))
    W[50:, P["jb"]] = 1.0
    return W


# ------------------------------------------------------------------ degerlendirme
def calistir(P, W, funding=0.0):
    eq, g = H.sim_portfoy(P["O"], P["C"], W, maliyet=MALIYET, funding=funding)
    # turnover serisi (sim_portfoy ile ayni agirlik surüklenmesi)
    O = P["O"]
    n = O.shape[0]
    to = np.zeros(len(g))
    w_prev = np.zeros(P["m"])
    for t in range(n - 2):
        w = np.nan_to_num(W[t]).copy()
        w[np.isnan(O[t + 1]) | np.isnan(O[t + 2])] = 0
        r = np.clip(np.nan_to_num((O[t + 2] - O[t + 1]) / O[t + 1]), -0.95, 5.0)
        d = np.abs(w - w_prev).sum()
        to[t] = d
        x = (w * r).sum() - d * MALIYET - np.abs(w).sum() * funding
        w_prev = w * (1 + r) / (1 + x) if x > -1 else w
    return eq, g, to


def rapor_bolum(P, g, to, bolum):
    gun = np.asarray(P["gun"])[:len(g)]
    a, b = H.BOLUM[bolum]
    msk = (gun >= a) & (gun < b)
    r = H.portfoy_rapor(None, g, P["gun"], bolum=bolum, kaynak="1d")
    if msk.sum() > 0:
        r["turnover_gun"] = round(float(to[msk].mean()), 4)
        r["turnover_yil"] = round(float(to[msk].mean() * 365), 1)
    return r


# ------------------------------------------------------------------ konfigurasyonlar
def grid():
    """Grid: H1 (+H3/H4 en iyi bolgede), H2. Isim -> build_W kwargs."""
    cfg = {}
    for K in (14, 30, 60, 90):
        for U in (30, 50, 100):
            for L in (10, 20):
                cfg[f"H1_K{K}_U{U}_L{L}"] = dict(K=K, U=U, L=L, mod="long")
    for K in (14, 30, 60, 90):
        for U in (50, 100):
            for L in (10,):
                cfg[f"H2_LS_K{K}_U{U}_L{L}"] = dict(K=K, U=U, L=L, mod="ls")
    return cfg


def ek_grid():
    """Tum H1 hucrelerine rejim filtresi (H3) ve/veya vol hedefleme (H4) ekle."""
    cfg = {}
    for ad, kw in grid().items():
        if not ad.startswith("H1_"):
            continue
        cfg[ad + "_REJ"] = dict(kw, rejim=True)
        cfg[ad + "_VOL"] = dict(kw, voltgt=True)
        cfg[ad + "_REJVOL"] = dict(kw, rejim=True, voltgt=True)
    return cfg


def tum_cfg():
    return dict(grid(), **ek_grid(), **kiyaslar(), **{"KIYAS_btc": dict(mod="btc")})


def kiyaslar():
    return {"KIYAS_ew30": dict(U=30, mod="ew"), "KIYAS_ew50": dict(U=50, mod="ew"),
            "KIYAS_ew100": dict(U=100, mod="ew")}


# ------------------------------------------------------------------ main
def tek_kosu(P, ad, kw, bolumler, kaydet=True):
    fund = FUNDING_LS if kw.get("mod") == "ls" else 0.0
    W = build_W_btc(P) if kw.get("mod") == "btc" else build_W(P, **kw)
    eq, g, to = calistir(P, W, funding=fund)
    out = {}
    for b in bolumler:
        r = rapor_bolum(P, g, to, b)
        out[b] = r
        if kaydet:
            H.kaydet(AILE, ad, dict(kw, funding=fund, maliyet=MALIYET, skip=SKIP,
                                    min_hist=MIN_HIST, rebal=kw.get("rebal", 7)), b, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="train", choices=["train", "ek", "valid", "kiyas"])
    ap.add_argument("--tek", default=None)
    ap.add_argument("--adaylar", default=None, help="virgullu isim listesi (valid fazi)")
    a = ap.parse_args()
    P = onhesap()
    print(f"panel: {P['m']} coin, {P['n']} gun", flush=True)

    if a.tek:
        cfg = {a.tek: tum_cfg()[a.tek]}; bol = ["train", "valid"]
    elif a.faz == "train":
        cfg = grid(); bol = ["train"]
    elif a.faz == "ek":
        cfg = ek_grid(); bol = ["train"]
    elif a.faz == "kiyas":
        cfg = dict(kiyaslar(), **{"KIYAS_btc": dict(mod="btc")}); bol = ["train", "valid"]
    else:
        tum = tum_cfg()
        cfg = {k: tum[k] for k in a.adaylar.split(",")}; bol = ["train", "valid"]

    son = {}
    for ad, kw in cfg.items():
        r = tek_kosu(P, ad, kw, bol)
        son[ad] = r
        tr = r.get("train", {}); va = r.get("valid", {})
        print(f"{ad:26s} TR sharpe={tr.get('sharpe')} cagr={tr.get('cagr')} dd={tr.get('maxdd')} "
              f"to={tr.get('turnover_gun')}" + (f" | VA sharpe={va.get('sharpe')} cagr={va.get('cagr')} "
              f"dd={va.get('maxdd')}" if va else ""), flush=True)
    with open(os.path.join(LAB, f"_son_{a.faz}.json"), "w", encoding="utf-8") as f:
        json.dump(son, f, indent=1, default=str)


if __name__ == "__main__":
    main()
