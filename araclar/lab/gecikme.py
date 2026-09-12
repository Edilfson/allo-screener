"""gecikme - UYGULAMA GERCEKCILIGI: gunluk kapanistan (00:00 UTC) sonra kac saat
icinde islem yapilirsa A2'nin alfasi korunuyor?

Aday A2 (tsmom/tsmom2): BTC/ETH/BNB gunluk MA50 long-only, esit agirlik,
hedef vol %30 (30g gerceklesen vol, cap 1.5x), gross <= 1.
Canli bot 01:10 UTC'de kosuyor -> pratikte ~+1h gecikme.

YENI STRATEJI ARAMASI YOK. Ayni kural, farkli UYGULAMA zamanlari.

Veri: harness.veri_1h (BTCUSDT/ETHUSDT/BNBUSDT, 2024-09..2025-12) -> resample('1d')
      ile gunluk kapanislar. Bolum: kaynak='1h' (train < 2025-07-01, valid 2025-07..12).
      HOLDOUT'a dokunulmaz (harness fiziksel keser).

Zaman kurgusu (ileriye bakma yok):
  gunluk bar k: acilis T_k = 00:00 UTC, kapanis T_k + 24h = T_{k+1}
  sinyal      : bar k'nin KAPANISINDA (yani T_{k+1} = 00:00 UTC aninda) belirlenir
  islem       : T_{k+1} + H saat -> o saatlik barin ACILIS fiyatindan
  getiri      : p_H[k] -> p_H[k+1] (tam 24 saat, gunde bir periyot)
  maliyet     : |dW| * 0.0007 (harness varsayilani; komisyon+kayma dahil)
  'anlik'     : H=0 referansi, gunluk KAPANIS fiyatindan (uygulanamaz ust sinir)

Fazlar:
  1 gecikme : H = anlik/0/1/2/4/8/12/24 saat, gunluk kontrol
  2 saatlik : MA50'yi 1h veriyle her saat guncelle (rolling 1200h), ayni gecikmeler
  3 olay    : sinyal degisim gunlerinde kapanis sonrasi 24h fiyat hareketi (bootstrap GA)
  4 ozet    : oneri + beklenen maliyet

Kullanim:
  PYTHONIOENCODING=utf-8 python gecikme.py --faz hepsi
  PYTHONIOENCODING=utf-8 python gecikme.py --faz gecikme
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "gecikme"
COINLER = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
MALIYET = 0.0007
MA_GUN = 50
VOLWIN = 30
HEDEF_VOL = 0.30
CAP = 1.5
VOL_TABAN = 0.10
SAAT = 3600_000
GUN_MS = 86400_000
GECIKMELER = [0, 1, 2, 4, 8, 12, 24]
KAYNAK = "1h"


# ------------------------------------------------------------------ yardimci
def roll_mean(A, N):
    """saga hizali (t dahil) N pencereli ortalama; pencerede nan varsa nan"""
    n, m = A.shape
    X = np.nan_to_num(A, nan=0.0); M = (~np.isnan(A)).astype(float)
    cs = np.cumsum(np.vstack([np.zeros((1, m)), X]), 0)
    cm = np.cumsum(np.vstack([np.zeros((1, m)), M]), 0)
    s = cs[N:] - cs[:-N]; c = cm[N:] - cm[:-N]
    out = np.full((n, m), np.nan)
    out[N - 1:] = np.where(c >= N, s / N, np.nan)
    return out


def roll_std(A, N):
    """ornek std (ddof=1), saga hizali, t dahil"""
    m1 = roll_mean(A, N); m2 = roll_mean(A * A, N)
    v = np.maximum(m2 - m1 * m1, 0.0)
    return np.sqrt(v * N / max(N - 1, 1))


def gunluk_sharpe(g):
    g = np.asarray(g, float)
    return float(g.mean() / g.std() * np.sqrt(365)) if len(g) > 1 and g.std() > 0 else 0.0


# ------------------------------------------------------------------ panel
_P = {}
def panel():
    """1h paneli + ondan turetilen gunluk panel. Tum semboller ayni zaman izgarasinda."""
    if _P:
        return _P
    ham = {s: H.veri_1h(s) for s in COINLER}
    ort = None
    for s in COINLER:
        t = ham[s][:, 0].astype(np.int64)
        ort = t if ort is None else np.intersect1d(ort, t)
    m = len(COINLER)
    hO = np.full((len(ort), m), np.nan); hC = np.full((len(ort), m), np.nan)
    for j, s in enumerate(COINLER):
        a = ham[s]; idx = np.searchsorted(a[:, 0].astype(np.int64), ort)
        hO[:, j] = a[idx, 1]; hC[:, j] = a[idx, 4]
    saat_idx = {int(t): i for i, t in enumerate(ort)}

    # gunluk (resample 1d, harness fonksiyonu; tam 24 bar olmayan gun dusuyor)
    dg = None
    dsz = {}
    for s in COINLER:
        d = H.resample(ham[s], "1d")
        dsz[s] = d
        t = d[:, 0].astype(np.int64)
        dg = t if dg is None else np.intersect1d(dg, t)
    dC = np.full((len(dg), m), np.nan)
    for j, s in enumerate(COINLER):
        d = dsz[s]; idx = np.searchsorted(d[:, 0].astype(np.int64), dg)
        dC[:, j] = d[idx, 4]
    _P.update(saat_ot=ort, hO=hO, hC=hC, saat_idx=saat_idx, gun_ot=dg, dC=dC)
    return _P


def fiyat_at(t_ms):
    """t_ms aninda ACILAN 1h barin acilis fiyatlari (yoksa nan)"""
    P = panel(); i = P["saat_idx"].get(int(t_ms))
    return P["hO"][i] if i is not None else np.full(len(COINLER), np.nan)


# ------------------------------------------------------------------ gunluk sinyal
def gunluk_agirlik(hedef=HEDEF_VOL, ma_gun=MA_GUN, volwin=VOLWIN, cap=CAP):
    """W[k,j]: gunluk bar k'nin KAPANISINDA bilinen hedef agirlik."""
    P = panel(); C = P["dC"]
    ma = roll_mean(C, ma_gun)
    R = np.full_like(C, np.nan); R[1:] = C[1:] / C[:-1] - 1
    vol = np.maximum(roll_std(R, volwin) * np.sqrt(365), VOL_TABAN)
    s = np.where(np.isnan(ma), 0.0, (C > ma).astype(float))
    hazir = ~np.isnan(ma) & ~np.isnan(vol)
    W = np.where(hazir, s / len(COINLER), 0.0)
    if hedef:
        W = W * np.clip(hedef / np.where(np.isnan(vol), 1e9, vol), 0, cap)
    gr = np.abs(W).sum(1, keepdims=True)
    W = np.where(gr > 1.0, W / np.maximum(gr, 1e-12), W)
    return W, s, hazir


def kos_gunluk(gecikme_h, W=None, anlik=False, maliyet=MALIYET, altut=False):
    """Gunluk kontrol + H saat gecikmeli uygulama. Kendi portfoy dongum.
    Return: dict(g=periyot getirileri, gun=periyot damgasi(=sinyal ani), ciro, ...)"""
    P = panel(); dg = P["gun_ot"]; dC = P["dC"]
    if W is None:
        W, _, _ = gunluk_agirlik()
    if altut:
        W = np.where(np.isnan(dC), 0.0, 1.0 / len(COINLER))
    n = len(dg)
    # p[k] = islem fiyati (k'nin kapanisinda verilen kararin uygulandigi fiyat)
    p = np.full((n, len(COINLER)), np.nan)
    for k in range(n):
        tc = int(dg[k]) + GUN_MS                     # k'nin kapanis ani (00:00 UTC)
        p[k] = dC[k] if anlik else fiyat_at(tc + gecikme_h * SAAT)
    g = []; gunler = []; ciro = []; wprev = np.zeros(len(COINLER))
    for k in range(n - 1):
        if np.any(np.isnan(p[k])) or np.any(np.isnan(p[k + 1])):
            continue
        w = np.nan_to_num(W[k])
        r = p[k + 1] / p[k] - 1
        c = np.abs(w - wprev).sum() * maliyet
        x = float((w * r).sum() - c)
        g.append(x); gunler.append(int(dg[k]) + GUN_MS); ciro.append(float(np.abs(w - wprev).sum()))
        wprev = w
    return dict(g=np.array(g), gun=np.array(gunler), ciro=np.array(ciro), W=W)


# ------------------------------------------------------------------ saatlik sinyal
def saatlik_agirlik(ma_saat=1200, vol_kaynak="gunluk", hedef=HEDEF_VOL, cap=CAP):
    """W[i,j]: saatlik bar i'nin KAPANISINDA bilinen hedef agirlik.
    MA = rolling ma_saat saatlik ortalama (1200h = 50 gun).
    vol_kaynak='gunluk' -> son KAPANMIS gunluk bardan vol30 (gunluk mod ile ayni olcek;
    boylece sadece SINYAL ZAMANLAMASI degisir). 'saatlik' -> 720h getiri vol'u."""
    P = panel(); ot = P["saat_ot"]; C = P["hC"]
    ma = roll_mean(C, ma_saat)
    s = np.where(np.isnan(ma), 0.0, (C > ma).astype(float))
    if vol_kaynak == "saatlik":
        Rh = np.full_like(C, np.nan); Rh[1:] = C[1:] / C[:-1] - 1
        vol = np.maximum(roll_std(Rh, 720) * np.sqrt(24 * 365), VOL_TABAN)
    else:
        dg = P["gun_ot"]; dC = P["dC"]
        Rd = np.full_like(dC, np.nan); Rd[1:] = dC[1:] / dC[:-1] - 1
        vold = np.maximum(roll_std(Rd, VOLWIN) * np.sqrt(365), VOL_TABAN)
        # saat i icin: kapanis ani ot[i]+1h; ondan ONCE veya TAM o anda kapanmis son gunluk bar
        kap = dg + GUN_MS
        idx = np.searchsorted(kap, ot + SAAT, side="right") - 1
        vol = np.where((idx >= 0)[:, None], vold[np.clip(idx, 0, len(kap) - 1)], np.nan)
    hazir = ~np.isnan(ma) & ~np.isnan(vol)
    W = np.where(hazir, s / len(COINLER), 0.0)
    W = W * np.clip(hedef / np.where(np.isnan(vol), 1e9, vol), 0, cap)
    gr = np.abs(W).sum(1, keepdims=True)
    W = np.where(gr > 1.0, W / np.maximum(gr, 1e-12), W)
    return W, s, hazir


def kos_saatlik(gecikme_h=1, vol_kaynak="gunluk", ma_saat=1200, maliyet=MALIYET):
    """Her saat kontrol, sinyal anindan gecikme_h saat SONRA islem.
    Bar i [t_i, t_i+1h): kapanisi t_i+1h = bar i+1'in ACILIS ani. Yani gecikme_h=0 ->
    bar i+1'in acilisi (sifir gecikme, ileriye bakma YOK); gecikme_h=1 -> bar i+2'nin acilisi.
    gecikme_h=-1 -> bar i'nin acilisi = ILERIYE BAKMA (sadece kontrol).
    Saatlik getiriler gunluk kovalara toplanir (gunluk mod ile ayni olcekte Sharpe icin)."""
    P = panel(); ot = P["saat_ot"]; hO = P["hO"]
    W, _, _ = saatlik_agirlik(ma_saat, vol_kaynak)
    n = len(ot)
    d = 1 + gecikme_h
    g = []; gunler = []; ciro = []; wprev = np.zeros(len(COINLER))
    for i in range(max(-d, 0), n - d - 1):
        a, b = i + d, i + d + 1
        if b >= n:
            break
        po, pn = hO[a], hO[b]
        if np.any(np.isnan(po)) or np.any(np.isnan(pn)):
            continue
        w = np.nan_to_num(W[i])
        r = pn / po - 1
        c = np.abs(w - wprev).sum() * maliyet
        g.append(float((w * r).sum() - c)); gunler.append(int(ot[a])); ciro.append(float(np.abs(w - wprev).sum()))
        wprev = w
    g = np.array(g); gunler = np.array(gunler); ciro = np.array(ciro)
    # gunluk kovalama (UTC gun)
    dkey = gunler // GUN_MS
    ug = np.unique(dkey)
    gd = np.array([float(np.prod(1 + g[dkey == k]) - 1) for k in ug])
    gd_ms = ug * GUN_MS
    cd = np.array([float(ciro[dkey == k].sum()) for k in ug])
    return dict(g=gd, gun=gd_ms, ciro=cd, g_saat=g, gun_saat=gunler)


# ------------------------------------------------------------------ rapor
def rap(res, bolum):
    g, gun = res["g"], res["gun"]
    r = H.portfoy_rapor(g, g, gun, bolum, KAYNAK)
    a, b = H.BOLUM_1H[bolum]
    m = (gun[:len(g)] >= a) & (gun[:len(g)] < b)
    if m.sum():
        yil = m.sum() / 365
        r["ciro_yil"] = round(float(res["ciro"][:len(g)][m].sum() / max(yil, 1e-9)), 2)
    return r


def kaydet_hepsi(ad, params, res, not_=""):
    out = {}
    for bl in ("train", "valid", "gelistirme"):
        r = rap(res, bl)
        out[bl] = r
        H.kaydet(AILE, ad, params, bl, r, not_)
    return out


# ------------------------------------------------------------------ FAZ 1
def faz_gecikme():
    W, s, hazir = gunluk_agirlik()
    sonuc = {}
    ref = kos_gunluk(0, W=W, anlik=True)
    sonuc["anlik"] = kaydet_hepsi("gunluk_anlik", dict(mod="gunluk", gecikme="anlik", fiyat="gunluk_kapanis"),
                                  ref, "uygulanamaz ust sinir")
    for h in GECIKMELER:
        r = kos_gunluk(h, W=W)
        sonuc[f"+{h}h"] = kaydet_hepsi(f"gunluk_g{h}h", dict(mod="gunluk", gecikme_h=h, fiyat="1h_acilis"), r)
    # kiyas: al-tut
    at = kos_gunluk(0, anlik=True, altut=True)
    sonuc["altut"] = kaydet_hepsi("altut_b3", dict(mod="altut", gecikme="anlik"), at, "kiyas")
    return sonuc, ref


# ------------------------------------------------------------------ FAZ 2
def faz_saatlik():
    sonuc = {}
    for h in GECIKMELER:
        r = kos_saatlik(h, "gunluk")
        sonuc[f"+{h}h"] = kaydet_hepsi(f"saatlik_g{h}h", dict(mod="saatlik", ma_saat=1200, gecikme_h=h,
                                                              vol="gunluk30"), r)
    r0 = kos_saatlik(-1, "gunluk")
    sonuc["ILERIYE_BAKMA_KONTROL_-1h"] = kaydet_hepsi(
        "saatlik_gEKSI1h_ILERIYEBAKMA", dict(mod="saatlik", ma_saat=1200, gecikme_h=-1, vol="gunluk30"), r0,
        "kapanis sinyali ile AYNI barin acilisindan islem = ileriye bakma; sadece kontrol, aday DEGIL")
    rv = kos_saatlik(1, "saatlik")
    sonuc["+1h_saatlikvol"] = kaydet_hepsi("saatlik_g1h_volsaat", dict(mod="saatlik", ma_saat=1200, gecikme_h=1,
                                                                       vol="saatlik720"), rv, "duyarlilik")
    # TANI (aday degil): maliyeti sifirlayarak 'ciro maliyeti' ile 'zamanlama gurultusu' ayristirmasi
    tani = {}
    for ad, res in (("gunluk_+1h", kos_gunluk(1, maliyet=0.0)), ("saatlik_+1h", kos_saatlik(1, "gunluk", maliyet=0.0))):
        tani[ad] = kaydet_hepsi("TANI_maliyetsiz_" + ad, dict(mod="tani", maliyet=0.0), res,
                                "TANI: maliyetsiz, aday DEGIL (ciro/gurultu ayristirmasi)")
    sonuc["_tani_maliyetsiz"] = tani
    return sonuc


# ------------------------------------------------------------------ FAZ 1b: eslesmis fark testi
def faz_istatistik():
    """Ayni sinyal gunleri uzerinde ESLESMIS fark: g_H - g_anlik.
    Yillik surukleme (%) ve Sharpe farki icin bootstrap GA95."""
    W, _, _ = gunluk_agirlik()
    ref = kos_gunluk(0, W=W, anlik=True)
    out = {}
    rng = np.random.default_rng(7)
    for h in GECIKMELER[1:] + [24]:
        r = kos_gunluk(h, W=W)
        n = min(len(ref["g"]), len(r["g"]))
        ok = np.isin(r["gun"][:n], ref["gun"][:n])
        a = ref["g"][:n][ok]; b = r["g"][:n][ok]
        d = b - a
        B = 4000; md = np.empty(B); sd = np.empty(B)
        for i in range(B):
            ix = rng.integers(0, len(d), len(d))
            md[i] = d[ix].mean()
            sd[i] = gunluk_sharpe(b[ix]) - gunluk_sharpe(a[ix])
        md.sort(); sd.sort()
        rec = dict(gun=int(len(d)),
                   yillik_surukleme_pct=round(float(d.mean() * 365 * 100), 2),
                   surukleme_ga95=[round(float(md[int(B * 0.025)]) * 365 * 100, 2),
                                   round(float(md[int(B * 0.975)]) * 365 * 100, 2)],
                   sharpe_fark=round(gunluk_sharpe(b) - gunluk_sharpe(a), 3),
                   sharpe_fark_ga95=[round(float(sd[int(B * 0.025)]), 3), round(float(sd[int(B * 0.975)]), 3)],
                   p_negatif=round(float((md < 0).mean()), 3))
        out[f"+{h}h"] = rec
        H.kaydet(AILE, f"eslesmis_fark_{h}h", dict(mod="eslesmis_fark", gecikme_h=h), "gelistirme",
                 dict(gun=rec["gun"], ort_net=rec["yillik_surukleme_pct"] / 100), "g_H - g_anlik")
    return out


# ------------------------------------------------------------------ FAZ 3
def faz_olay():
    """Sinyal degisim gunlerinde kapanis sonrasi h saatlik hareket.
    giris (0->1): pozitif hareket = gecikme ZARARLI (kaciran getiri)
    cikis (1->0): pozitif hareket = gecikme KARLI (daha yuksekten satis)
    isaretli = -r (giris) / +r (cikis) -> negatif ortalama = gecikme maliyeti"""
    P = panel(); dg = P["gun_ot"]; dC = P["dC"]
    W, s, hazir = gunluk_agirlik()
    saatler = [1, 2, 4, 8, 12, 24]
    olay = {"giris": {h: [] for h in saatler}, "cikis": {h: [] for h in saatler}}
    agir = {"giris": [], "cikis": []}
    n = len(dg)
    for k in range(1, n):
        if not hazir[k].all() or not hazir[k - 1].all():
            continue
        tc = int(dg[k]) + GUN_MS
        for j in range(len(COINLER)):
            if s[k, j] == s[k - 1, j]:
                continue
            tip = "giris" if s[k, j] > s[k - 1, j] else "cikis"
            p0 = dC[k, j]
            ok = True; tmp = {}
            for h in saatler:
                ph = fiyat_at(tc + h * SAAT)[j]
                if np.isnan(ph):
                    ok = False; break
                tmp[h] = float(ph / p0 - 1)
            if not ok:
                continue
            for h in saatler:
                olay[tip][h].append(tmp[h])
            agir[tip].append(float(W[k, j] if tip == "giris" else W[k - 1, j]))
    out = {"n_giris": len(olay["giris"][1]), "n_cikis": len(olay["cikis"][1]),
           "ort_agirlik_giris": round(float(np.mean(agir["giris"])), 4) if agir["giris"] else None,
           "ort_agirlik_cikis": round(float(np.mean(agir["cikis"])), 4) if agir["cikis"] else None,
           "saatler": {}}
    yil = (dg[-1] - dg[0]) / (365 * GUN_MS)
    wg = np.array(agir["giris"], float); wc = np.array(agir["cikis"], float)
    rng = np.random.default_rng(11)
    for h in saatler:
        gi = np.array(olay["giris"][h]); ci = np.array(olay["cikis"][h])
        isaretli = np.concatenate([-gi, ci]) if len(gi) + len(ci) else np.array([])
        ga = H._ga(isaretli) if len(isaretli) >= 5 else (None, None)
        # portfoye katki: agirlikla olcekli, olay basi (sabit-agirlik kosularinda fark SADECE
        # olaylardan gelir -> teleskoplama; bu yuzden bu tahminci gunluk esleseme gore cok daha dar)
        katki = np.concatenate([-gi * wg, ci * wc])
        B = 4000; mk = np.sort([katki[rng.integers(0, len(katki), len(katki))].mean() for _ in range(B)])
        olc = len(katki) / max(yil, 1e-9)
        out["saatler"][f"+{h}h"] = dict(
            giris_ort=round(float(gi.mean()), 5) if len(gi) else None,
            giris_medyan=round(float(np.median(gi)), 5) if len(gi) else None,
            cikis_ort=round(float(ci.mean()), 5) if len(ci) else None,
            cikis_medyan=round(float(np.median(ci)), 5) if len(ci) else None,
            isaretli_ort=round(float(isaretli.mean()), 5) if len(isaretli) else None,
            isaretli_ga95=[round(v, 5) for v in ga] if ga[0] is not None else None,
            yillik_etki_pct=round(100 * float(katki.mean()) * olc, 2),
            yillik_etki_ga95=[round(100 * float(mk[int(B * .025)]) * olc, 2),
                              round(100 * float(mk[int(B * .975)]) * olc, 2)],
            p_zararli=round(float((mk < 0).mean()), 3))
    H.kaydet(AILE, "olay_analizi", dict(mod="olay", coinler=COINLER), "gelistirme",
             dict(n=out["n_giris"] + out["n_cikis"], ort_net=out["saatler"]["+1h"]["isaretli_ort"]),
             "sinyal degisim gunlerinde kapanis sonrasi hareket")
    return out


# ------------------------------------------------------------------ FAZ 4: ozet
def faz_ozet(D):
    g1, g1b, g2, g3 = D["faz1_gunluk"], D["faz1b_eslesmis"], D["faz2_saatlik"], D["faz3_olay"]
    kis = lambda v: {b: dict(sharpe=v[b].get("sharpe"), maxdd=v[b].get("maxdd"), toplam=v[b].get("toplam"),
                             cagr=v[b].get("cagr"), gun=v[b].get("gun"), ciro_yil=v[b].get("ciro_yil"))
                     for b in ("train", "valid", "gelistirme")}
    kayit = [k for k in H.defter_oku() if k.get("aile") == AILE]
    n = len({k["ad"] for k in kayit})       # tekil konfigurasyon (defterde tekrar kosular da var)
    tablo = {}
    for k, v in g1.items():
        if k == "altut":
            continue
        tablo[k] = kis(v)
        if k in g1b:
            tablo[k]["eslesmis_fark_vs_anlik"] = g1b[k]
        if k in g3["saatler"]:
            tablo[k]["olay_bazli_yillik_etki_pct"] = g3["saatler"][k]["yillik_etki_pct"]
            tablo[k]["olay_bazli_ga95"] = g3["saatler"][k]["yillik_etki_ga95"]
    tablo2 = {k: kis(v) for k, v in g2.items() if not k.startswith("_")}
    tani = {k: kis(v) for k, v in g2["_tani_maliyetsiz"].items()}
    return dict(
        aile=AILE, denenen=n, kaynak="1h -> resample 1d",
        veri="BTCUSDT/ETHUSDT/BNBUSDT 1h, 2024-09-13..2025-12-31 (475 gunluk bar; MA50 sonrasi 425 etkin gun)",
        bolum="kaynak='1h': train < 2025-07-01 (290 gun) | valid 2025-07..12 (184 gun) | gelistirme = ikisi",
        strateji="A2: gunluk MA50 long-only, esit agirlik, hedef vol %30 (30g, cap 1.5), gross<=1, maliyet 0.0007*|dW|",
        kiyas_altut=kis(g1["altut"]),
        capraz_kontrol="kendi portfoy dongum H=0 => harness.sim_portfoy ile birebir ayni "
                       "(gelistirme Sharpe 1.61, toplam 0.440, maxDD 0.128)",
        soru1_gunluk_kontrol_gecikme=tablo,
        soru2_saatlik_kontrol=tablo2,
        soru2_tani_maliyetsiz=tani,
        soru3_olay=g3,
        en_iyi_3=[
            dict(ad="+1h (01:00-01:10 UTC) - ONERILEN", params=dict(mod="gunluk_kontrol", gecikme_h=1),
                 train=kis(g1["+1h"])["train"], valid=kis(g1["+1h"])["valid"],
                 komsular="anlik 1.61 / +1h 1.62 / +2h 1.51 / +4h 1.60 / +8h 1.61 / +12h 1.71 (gelistirme Sharpe). "
                          "+1h ile anlik arasindaki fark olculemiyor: eslesmis surukleme -0.19%/yil GA95 [-10.2,+9.5], "
                          "olay-bazli -0.43%/yil GA95 [-1.5,+0.7]."),
            dict(ad="+2h..+12h - kabul edilebilir", params=dict(mod="gunluk_kontrol", gecikme_h="2..12"),
                 train=kis(g1["+4h"])["train"], valid=kis(g1["+4h"])["valid"],
                 komsular="Bu bandin icinde Sharpe 1.51-1.71 arasinda salsalaniyor; sirasi monoton DEGIL "
                          "(+2h en dusuk, +12h en yuksek) -> fark gurultu. Olay-bazli maliyet -2.0%..+1.3%/yil."),
            dict(ad="+24h (tam gun gecikme) - KACIN", params=dict(mod="gunluk_kontrol", gecikme_h=24),
                 train=kis(g1["+24h"])["train"], valid=kis(g1["+24h"])["valid"],
                 komsular="Tek belirgin bozulma: gelistirme Sharpe 1.61->1.36 (-0.25), toplam 0.44->0.37, "
                          "olay-bazli -4.3%/yil GA95 [-10.7,+1.6], p(zararli)=0.92. 1d kaynakli tsmom2 sonucu "
                          "(-0.30 Sharpe) ile tutarli - bagimsiz veri, ayni yon."),
        ],
        dusenler=[
            "SAATLIK kontrol (rolling 1200h MA): +1h gecikmede gelistirme Sharpe 1.38 vs gunluk kontrolun 1.62; "
            "toplam 0.365 vs 0.438. Ciro 62/yil vs 15.7/yil (yillik maliyet %4.3 vs %1.1).",
            "Saatlik kontrol + saatlik vol (720h, cap 1.5): 1.43 - vol kaynagini degistirmek acigi kapatmiyor.",
            "Saatlik kontrolde sinyali BEKLETMEK iyilestiriyor (+0h 1.27 -> +12h 1.62 -> +24h 1.62): saatlik "
            "tazelik bir deger degil, whipsaw kaynagi.",
            "Gunluk +2h dilimi (+1h ve +4h'ten kotu, 1.51): tek basina anlamsiz, ornek gurultusu "
            "(eslesmis dSharpe -0.10, GA95 [-0.80, +0.62]).",
        ],
        ogrenilen=[
            "Gunluk kapanistan sonraki ILK 12 SAAT icinde islem yapmak pratik olarak bedava: +1h/+4h/+8h/+12h "
            "gelistirme Sharpe 1.62/1.60/1.61/1.71 vs anlik 1.61. Alfa saatler icinde bozulmuyor; sadece TAM GUN "
            "gecikmede (1.36) kayboluyor. Yani kritik esik 'ilk saatler' degil, 'ayni gun icinde'.",
            "Ayni sinyalle farkli fiyat izgaralarinin farki SADECE sinyal degisim gunlerinden gelir (aradaki "
            "gunler teleskoplanir). 75 olayin (37 giris / 38 cikis) uzerinden olculen olay-bazli maliyet, "
            "gunluk eslesmis farktan ~10 kat dar GA veriyor ve gozlenen equity farkini yeniden uretiyor "
            "(+2h: tahmin -2.6pp vs gozlenen -2.9pp; +24h: -5.6pp vs -7.3pp).",
            "Kapanis sonrasi hareketin YONU sistematik degil: giris gunlerinde +1h ortalama hareket +%0.07 "
            "(aleyhte), cikis gunlerinde +%0.01 (lehte); isaretli ortalama -%0.03, GA95 sifiri iceriyor. "
            "MA50 kesisimi bir 'anons' olayi degil - kapanistan sonra kovalanacak bir hareket yok.",
            "SAATLIK sinyal tazeligi ZARARLI: ayni +1h gecikmede 1.62 -> 1.38. Ayrisim (maliyetsiz TANI kosusu: "
            "gunluk 1.68, saatlik 1.61): 0.17 Sharpe fazladan CIRODAN (62 vs 15.7 islem/yil), 0.07 Sharpe "
            "zamanlama gurultusundan. Gunluk kapanis, MA50 icin bir gecikme degil bir FILTRE.",
            "1h penceresi kisa (425 gun, 75 olay) ve rejim elverisli (valid Sharpe 2.1-2.2, 1d kaynakli 2025 "
            "valid 1.39'un uzerinde). Seviyelere degil, gecikmeler ARASINDAKI farka guven; o fark +24h disinda "
            "istatistiksel olarak sifir.",
        ],
        uyarilar=[
            "Pencere KISA: 1h veri 2024-09'da basliyor, MA50 sonrasi 425 etkin gun ve sadece 75 sinyal degisim "
            "olayi. Sharpe SEVIYELERI (1.6 / valid 2.2) 1d kaynakli calismanin cok uzerinde - rejim etkisi; "
            "bu ailenin ciktisi seviye degil, gecikmeler ARASINDAKI FARK.",
            "Gecikme egrisi MONOTON DEGIL (+2h en dusuk, +12h en yuksek). Bu, gerçek bir 'en iyi saat' degil; "
            "gun ici zamanlamanin sinyal olmadiginin kanitidir. +12h'i secmeyin - overfit olur.",
            "Saatlik moddaki 'ILERIYE_BAKMA_KONTROL_-1h' satiri (Sharpe 3.43) bilerek eklendi: sinyal barinin "
            "kendi acilisindan islem yapmak alfayi 2 katina cikariyor. Bir bot bunu yanlislikla yaparsa "
            "backtest'i saglama alma refleksi icin referans.",
            "Maliyet harness varsayilani 0.0007*|dW| (komisyon+kayma birlikte); degistirilmedi. "
            "Ayri bir fiyat-seviyesi kaymasi EKLENMEDI - bu maliyet zaten kaymayi iceriyor.",
            "HOLDOUT'a dokunulmadi (LAB_HOLDOUT ayarlanmadi; veri 2025-12-31'de fiziksel kesik).",
        ],
        oneri="1) Canli botu 01:10 UTC'de birakin - dogru karar. Beklenen gecikme maliyeti: yillik -0.4% "
              "(GA95 [-1.5%, +0.7%]), Sharpe etkisi +0.01 (GA95 [-0.56, +0.55]) yani olculemiyor. "
              "2) Operasyonel tolerans genis: bot kacsa bile ayni gun 12:00 UTC'ye kadar telafi edilirse "
              "beklenen ek maliyet < %2/yil. Kirmizi cizgi bir sonraki gunluk kapanis: tam gun kayarsa "
              "-4..-6%/yil ve -0.25 Sharpe. Bu yuzden bot icin 'kacirilan gun' alarmi + ayni gun icinde "
              "telafi kosusu (orn. 13:00 UTC yedek tetik) yeterli; saniye hassasiyeti gereksiz. "
              "3) Saatlik yeniden degerlendirmeyi UYGULAMAYIN - gunluk kapanis kontrolunde kalin. "
              "4) Bir sonraki ajan icin: bu ailede acik soru kalmadi; asil olculmemis uygulama riski "
              "SLIPAJ/LIKIDITE degil (b3'te ihmal), pozisyon boyutunun vol tahminine duyarliligi ve "
              "borsa kesintisi/emir reddi senaryolari. Onun disinda A2 uygulama tarafinda saglam.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="hepsi", choices=["hepsi", "gecikme", "saatlik", "olay", "istat", "ozet"])
    a = ap.parse_args()
    if a.faz == "ozet":            # simulasyonu tekrar kosmadan _gecikme_ham.json'dan rapor uret
        D = json.load(open(os.path.join(LAB, "_gecikme_ham.json"), encoding="utf-8"))
        p = os.path.join(LAB, "sonuc_gecikme.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(faz_ozet(D), f, ensure_ascii=False, indent=1, default=str)
        print("sonuc ->", p); return
    P = panel()
    print("1h bar:", len(P["saat_ot"]), "| gunluk bar:", len(P["gun_ot"]),
          datetime.fromtimestamp(P["gun_ot"][0] / 1e3, timezone.utc).strftime("%Y-%m-%d"),
          "->", datetime.fromtimestamp(P["gun_ot"][-1] / 1e3, timezone.utc).strftime("%Y-%m-%d"))
    D = {}
    if a.faz in ("hepsi", "gecikme"):
        D["faz1_gunluk"], _ = faz_gecikme()
        for k, v in D["faz1_gunluk"].items():
            print(f"  G {k:>8} train S={v['train'].get('sharpe')} valid S={v['valid'].get('sharpe')} "
                  f"tum S={v['gelistirme'].get('sharpe')} dd={v['gelistirme'].get('maxdd')} "
                  f"top={v['gelistirme'].get('toplam')}")
    if a.faz in ("hepsi", "istat"):
        D["faz1b_eslesmis"] = faz_istatistik()
        for k, v in D["faz1b_eslesmis"].items():
            print(f"  F {k:>5} surukleme {v['yillik_surukleme_pct']:+.2f}%/yil GA95 {v['surukleme_ga95']} | "
                  f"dSharpe {v['sharpe_fark']:+.2f} GA95 {v['sharpe_fark_ga95']}")
    if a.faz in ("hepsi", "saatlik"):
        D["faz2_saatlik"] = faz_saatlik()
        for k, v in D["faz2_saatlik"].items():
            if k.startswith("_"):
                continue
            print(f"  S {k:>16} train S={v['train'].get('sharpe')} valid S={v['valid'].get('sharpe')} "
                  f"tum S={v['gelistirme'].get('sharpe')} dd={v['gelistirme'].get('maxdd')} "
                  f"top={v['gelistirme'].get('toplam')} ciro={v['gelistirme'].get('ciro_yil')}")
    if a.faz in ("hepsi", "olay"):
        D["faz3_olay"] = faz_olay()
        print("  olay:", json.dumps(D["faz3_olay"]["saatler"], ensure_ascii=False)[:600])
    with open(os.path.join(LAB, "_gecikme_ham.json"), "w", encoding="utf-8") as f:
        json.dump(D, f, ensure_ascii=False, indent=1, default=str)
    print("ham ->", os.path.join(LAB, "_gecikme_ham.json"))
    if a.faz == "hepsi":
        p = os.path.join(LAB, "sonuc_gecikme.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(faz_ozet(D), f, ensure_ascii=False, indent=1, default=str)
        print("sonuc ->", p)


if __name__ == "__main__":
    main()
