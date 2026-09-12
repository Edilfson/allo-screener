"""tsmom2 - A1/A2'nin SECIM YANLILIGI ve UYGULAMA testleri (kaynak=1d, portfoy bazli).

Onceki ajan (tsmom): A1 = BTC/ETH/BNB gunluk MA50 long-only esit agirlik,
                     A2 = ayni sinyal + hedef vol %30 (cap 1.5).
Acik soru: b3 sepeti GECMISE BAKARAK secildi (BNB 2018'de 'ilk 3' degildi).

Bu aile YENI PARAMETRE ARAMAZ. tsmom.py fonksiyonlarini import eder, ayni MA50(+vt30)
kuralini farkli SEPETLERDE ve farkli UYGULAMA varsayimlarinda kosar.

  1) Sepet duyarliligi: {BTC}, {BTC,ETH}, {BTC,ETH,BNB}, {BTC,ETH,XRP}, {BTC,ETH,LTC},
     {BTC,ETH,XRP,LTC,BCH} ve YURUYEN sepet (her ay basi, gecmis 90g USDT hacmi ilk 3).
  2) Yil bazinda strateji vs sepet al-tut (getiri + maxDD), 2018..2025.
  3) Uygulama: haftalik (pazartesi) rebalans, sinyal gecikmesi 1 gun (t+2 acilis),
     maliyet 0.1% / 0.2% tek yon.
  4) Sonuc gunlugu: en kotu 5 dusus + sureleri, ciro/yil, islem sayisi/yil.

TRAIN ve VALID sadece RAPORLANIR; secim yapilmaz. HOLDOUT'a dokunulmaz.

Kullanim:
  PYTHONIOENCODING=utf-8 python tsmom2.py --faz hepsi     # hesapla + sonuc_tsmom2.json
  PYTHONIOENCODING=utf-8 python tsmom2.py --faz sepet     # sadece 1. soru
"""
import argparse, json, os, sys
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import tsmom as T

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "tsmom2"
MALIYET_VARSAYILAN = 0.0007          # harness varsayilani (tek yon, birim ciro)

# 2018 basinda BILINEBILIR sepetler (ileriye bakma yok) + orijinal (gecmise bakan) sepet
SEPETLER = {
    "s1_btc":       ["BTCUSDT"],
    "s2_btc_eth":   ["BTCUSDT", "ETHUSDT"],
    "s3_orij_bnb":  ["BTCUSDT", "ETHUSDT", "BNBUSDT"],          # A1/A2'nin sepeti (gecmise bakan secim)
    "s3_2018_xrp":  ["BTCUSDT", "ETHUSDT", "XRPUSDT"],          # 2018 piyasa degeri ilk 3
    "s3_ltc":       ["BTCUSDT", "ETHUSDT", "LTCUSDT"],
    "s5_2018":      ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT"],   # 2018 ilk 5
    "yuruyen_top3": None,                                       # aylik hacim siralamasi
}


# ------------------------------------------------------------------ evren
def _ay_indeks(gun):
    return np.array([(lambda d: d.year * 12 + d.month)(datetime.fromtimestamp(g / 1e3, timezone.utc)) for g in gun])


_AY = {}
def evren_maske2(tam, cfg):
    """t gunu KAPANISINDA bilinen secilebilir semboller. Yuruyen sepet: her ayin ilk gununde,
    bir ONCEKI gunun 90g ortalama dolar hacmine gore ilk N. (ileriye bakma yok)"""
    P = T.panel(); syms, Cf, dv, gun = P["syms"], P["Cf"], P["dv"], P["gun"]
    vol = T.volat(cfg.get("volwin", 30))
    canli = tam & ~np.isnan(Cf) & ~np.isnan(vol)
    if cfg["evren"] == "sabit":
        col = np.zeros(len(syms), bool)
        for s in cfg["liste"]:
            if s in syms:
                col[syms.index(s)] = True
        return canli & col[None, :]
    # ---- yuruyen (aylik yeniden secim)
    N = cfg["top"]
    k = ("aylik", N, cfg.get("volwin", 30))
    if k in _AY:
        return _AY[k] & canli
    if "ay" not in P:
        P["ay"] = _ay_indeks(gun)
    ay = P["ay"]
    n, m = Cf.shape
    M = np.zeros((n, m), bool); sec = []
    for t in range(n):
        if t == 0 or ay[t] != ay[t - 1]:
            src = max(t - 1, 0)
            sk = np.where(canli[src] & ~np.isnan(dv[src]), dv[src], -np.inf)
            order = np.argsort(-sk)[:N]
            sec = [int(j) for j in order if np.isfinite(sk[j])]
        if sec:
            M[t, sec] = True
    _AY[k] = M
    return M & canli


def sepet_tarihce(N=3, volwin=30):
    """Yuruyen sepetin ay ay bilesimi (rapor icin)."""
    P = T.panel(); gun, syms = P["gun"], P["syms"]
    frac, tam = T.oylar([50], [], 0.0)
    M = evren_maske2(tam, dict(evren="aylik_top", top=N, volwin=volwin))
    if "ay" not in P:
        P["ay"] = _ay_indeks(gun)
    ay = P["ay"]; b = P["bas"]
    out = []; onceki = None
    for t in range(b, len(gun)):
        if t > b and ay[t] == ay[t - 1]:
            continue
        cur = tuple(sorted(syms[j].replace("USDT", "") for j in np.flatnonzero(M[t])))
        d = datetime.fromtimestamp(gun[t] / 1e3, timezone.utc).strftime("%Y-%m")
        if cur != onceki:
            out.append([d, list(cur)])
            onceki = cur
    return out


# ------------------------------------------------------------------ agirlik
def _pazartesi(gun):
    d = (np.asarray(gun) // 86400_000).astype(np.int64)
    return (d % 7) == 4          # 1970-01-01 = Persembe -> Pazartesi = gun 4


def agirliklar2(cfg):
    """tsmom.agirliklar ile AYNI sinyal mantigi (MA + opsiyonel vol hedefleme),
    ek olarak: al-tut modu, haftalik rebalans, sinyal gecikmesi."""
    P = T.panel(); gun = P["gun"]
    vol = T.volat(cfg.get("volwin", 30))
    frac, tam = T.oylar(cfg["ma"], cfg["ret"], cfg["band"])
    um = evren_maske2(tam, cfg)
    s = np.ones_like(frac) if cfg.get("altut") else np.nan_to_num(frac)
    s = np.where(um, s, 0.0)
    K = um.sum(1)
    b = np.where(um, 1.0 / np.maximum(K, 1)[:, None], 0.0)
    W = b * s
    if cfg.get("hedef") and not cfg.get("altut"):
        W = W * np.clip(cfg["hedef"] / np.nan_to_num(vol, nan=1e9), 0, cfg.get("cap", T.VOL_UST))
    gross = np.abs(W).sum(1, keepdims=True)
    W = np.where(gross > 1.0, W / np.maximum(gross, 1e-12), W)
    if cfg.get("rebalans") == "haftalik":
        pzt = _pazartesi(gun)
        out = np.zeros_like(W); cur = np.zeros(W.shape[1])
        for t in range(len(W)):
            if pzt[t] or t == 0:
                cur = W[t]
            out[t] = cur
        W = out
    g = int(cfg.get("gecikme", 0))
    if g:
        W = np.vstack([np.zeros((g, W.shape[1])), W[:-g]])
    return W


def C2(ad, **kw):
    d = dict(ad=ad, evren="sabit", liste=["BTCUSDT"], top=3, ma=[50], ret=[], band=0.0,
             hedef=None, cap=T.VOL_UST, volwin=30, altut=False, rebalans="gunluk", gecikme=0)
    d.update(kw)
    return d


def sepet_cfg(ad, **kw):
    L = SEPETLER[ad]
    if L is None:
        return C2(ad, evren="aylik_top", top=3, **kw)
    return C2(ad, evren="sabit", liste=L, **kw)


# ------------------------------------------------------------------ kosu / olcum
def kos2(cfg, bolumler=("train", "valid"), maliyet=MALIYET_VARSAYILAN, kaydet=True, ad=None, not_=""):
    P = T.panel(); b = P["bas"]
    W = agirliklar2(cfg)
    eq, g = H.sim_portfoy(P["O"][b:], P["C"][b:], W[b:], maliyet=maliyet)
    gun = P["gun"][b:]
    ciro, giris = akis(W[b:], gun)
    out = {}
    for bl in bolumler:
        r = H.portfoy_rapor(eq, g, gun, bl)
        a, bb = H.BOLUM[bl]
        m = (gun[:len(g)] >= a) & (gun[:len(g)] < bb)
        yil = len(g[m]) / 365 if m.sum() else 1
        r["ciro_yil"] = round(float(ciro[:len(g)][m].sum() / max(yil, 1e-9)), 2)
        r["islem_yil"] = round(float(giris[:len(g)][m].sum() / max(yil, 1e-9)), 1)
        r["ort_gross"] = round(float(np.abs(W[b:][:len(g)][m]).sum(1).mean()), 3)
        out[bl] = r
        if kaydet:
            H.kaydet(AILE, ad or cfg["ad"], dict(cfg, maliyet=maliyet), bl, r, not_)
    out["_eq"] = eq; out["_g"] = g; out["_gun"] = gun; out["_W"] = W[b:]
    return out


def akis(Wb, gun):
    """gunluk ciro (sum|dW|) ve gunluk yeni-pozisyon sayisi"""
    prev = np.vstack([np.zeros((1, Wb.shape[1])), Wb[:-1]])
    ciro = np.abs(Wb - prev).sum(1)
    giris = ((Wb > 1e-9) & (prev <= 1e-9)).sum(1).astype(float)
    return ciro, giris


def yillik(g, gun):
    """her yil icin getiri ve o yil icindeki maxDD"""
    g = np.asarray(g); gun = np.asarray(gun)[:len(g)]
    yl = {}
    for t, x in zip(gun, g):
        y = datetime.fromtimestamp(t / 1e3, timezone.utc).year
        yl.setdefault(y, []).append(x)
    out = {}
    for y, v in sorted(yl.items()):
        v = np.array(v); eq = np.cumprod(1 + v)
        pk = np.maximum.accumulate(eq)
        out[y] = dict(getiri=round(float(eq[-1] - 1), 3), maxdd=round(float(((pk - eq) / pk).max()), 3), gun=len(v))
    return out


def dd_dagilim(g, gun, k=5):
    """en kotu k dusus: derinlik, tepe/dip/toparlanma tarihi, sure(gun), toparlanma(gun)"""
    g = np.asarray(g); gun = np.asarray(gun)[:len(g)]
    eq = np.concatenate([[1.0], np.cumprod(1 + g)])
    t_ms = np.concatenate([[gun[0]], gun])
    ep = []; pk = eq[0]; pk_i = 0; dip = eq[0]; dip_i = 0; icinde = False
    for i in range(1, len(eq)):
        if eq[i] >= pk:
            if icinde:
                ep.append((pk_i, dip_i, i, (pk - dip) / pk)); icinde = False
            pk = eq[i]; pk_i = i; dip = eq[i]; dip_i = i
        else:
            if not icinde:
                icinde = True; dip = eq[i]; dip_i = i
            if eq[i] < dip:
                dip = eq[i]; dip_i = i
    if icinde:
        ep.append((pk_i, dip_i, None, (pk - dip) / pk))
    ep.sort(key=lambda x: -x[3])
    f = lambda i: datetime.fromtimestamp(t_ms[i] / 1e3, timezone.utc).strftime("%Y-%m-%d")
    out = []
    for p, d, r, dep in ep[:k]:
        out.append(dict(derinlik=round(float(dep), 3), tepe=f(p), dip=f(d),
                        dususte_gun=int(d - p), toparlanma=f(r) if r else "toparlanmadi",
                        toparlanma_gun=int(r - d) if r else None))
    return out


# ------------------------------------------------------------------ fazlar
def faz_sepet(kaydet=True):
    """1. soru: ayni kural (MA50 ve MA50+vt30) yedi farkli sepette + her sepetin al-tutu"""
    tablo = []
    for ad in SEPETLER:
        satir = dict(ad=ad, coinler=SEPETLER[ad] or "aylik ilk-3 (90g USDT hacmi)")
        for etiket, kw in (("ma50", {}), ("ma50_vt30", dict(hedef=0.30)), ("altut", dict(altut=True))):
            c = sepet_cfg(ad, **kw)
            r = kos2(c, ("train", "valid"), kaydet=kaydet, ad=f"{ad}__{etiket}")
            satir[etiket] = {b: {k: v for k, v in r[b].items() if not k.startswith("_")} for b in ("train", "valid")}
        tablo.append(satir)
        p = lambda e, b: satir[e][b]
        print(f"{ad:<14} MA50 tr S={p('ma50','train')['sharpe']:>5} dd={p('ma50','train')['maxdd']:<6} "
              f"va S={p('ma50','valid')['sharpe']:>5} dd={p('ma50','valid')['maxdd']:<6} | "
              f"VT30 tr S={p('ma50_vt30','train')['sharpe']:>5} dd={p('ma50_vt30','train')['maxdd']:<6} "
              f"va S={p('ma50_vt30','valid')['sharpe']:>5} dd={p('ma50_vt30','valid')['maxdd']:<6} | "
              f"altut tr S={p('altut','train')['sharpe']:>5} va S={p('altut','valid')['sharpe']:>5}")
    return tablo


def faz_yil(kaydet=True):
    """2. soru: yil bazinda strateji vs sepet al-tut"""
    out = {}
    for ad in SEPETLER:
        rs = kos2(sepet_cfg(ad, hedef=0.30), ("train",), kaydet=kaydet, ad=f"{ad}__ma50_vt30", not_="yil-analizi")
        r1 = kos2(sepet_cfg(ad), ("train",), kaydet=kaydet, ad=f"{ad}__ma50", not_="yil-analizi")
        rb = kos2(sepet_cfg(ad, altut=True), ("train",), kaydet=kaydet, ad=f"{ad}__altut", not_="yil-analizi")
        ys, y1, yb = yillik(rs["_g"], rs["_gun"]), yillik(r1["_g"], r1["_gun"]), yillik(rb["_g"], rb["_gun"])
        yl = {}
        for y in sorted(set(ys) & set(yb)):
            yl[y] = dict(vt30=[ys[y]["getiri"], ys[y]["maxdd"]], ma50=[y1[y]["getiri"], y1[y]["maxdd"]],
                         altut=[yb[y]["getiri"], yb[y]["maxdd"]], gun=yb[y]["gun"])
        kaz_v = sum(1 for y in yl if yl[y]["vt30"][0] > yl[y]["altut"][0])
        kaz_1 = sum(1 for y in yl if yl[y]["ma50"][0] > yl[y]["altut"][0])
        dd_v = sum(1 for y in yl if yl[y]["vt30"][1] < yl[y]["altut"][1])
        dd_1 = sum(1 for y in yl if yl[y]["ma50"][1] < yl[y]["altut"][1])
        out[ad] = dict(yillar=yl, yil_sayisi=len(yl),
                       getiri_gecti=dict(ma50=kaz_1, vt30=kaz_v), dd_daha_dusuk=dict(ma50=dd_1, vt30=dd_v))
        print(f"{ad:<14} yil={len(yl)}  getiri>altut: ma50 {kaz_1}/{len(yl)} vt30 {kaz_v}/{len(yl)}  "
              f"dd<altut: ma50 {dd_1}/{len(yl)} vt30 {dd_v}/{len(yl)}")
    return out


UYG = [("temel", {}, MALIYET_VARSAYILAN),
       ("haftalik", dict(rebalans="haftalik"), MALIYET_VARSAYILAN),
       ("gecikme1", dict(gecikme=1), MALIYET_VARSAYILAN),
       ("haftalik_gecikme1", dict(rebalans="haftalik", gecikme=1), MALIYET_VARSAYILAN),
       ("maliyet10bp", {}, 0.0010),
       ("maliyet20bp", {}, 0.0020),
       ("haftalik_maliyet20bp", dict(rebalans="haftalik"), 0.0020),
       ("gecikme1_maliyet20bp", dict(gecikme=1), 0.0020)]


def faz_uygulama(sepetler=("s3_orij_bnb", "s3_2018_xrp", "yuruyen_top3"), kaydet=True):
    """3. soru: rebalans sikligi, sinyal gecikmesi, maliyet dayanikliligi"""
    out = {}
    for sp in sepetler:
        for kural, kw0 in (("ma50", {}), ("ma50_vt30", dict(hedef=0.30))):
            key = f"{sp}__{kural}"; out[key] = {}
            for etiket, kw, mal in UYG:
                c = sepet_cfg(sp, **dict(kw0, **kw))
                r = kos2(c, ("train", "valid"), maliyet=mal, kaydet=kaydet, ad=f"{key}__{etiket}")
                out[key][etiket] = {b: {k: v for k, v in r[b].items() if not k.startswith("_")}
                                    for b in ("train", "valid")}
                t, v = out[key][etiket]["train"], out[key][etiket]["valid"]
                print(f"{key:<26} {etiket:<22} tr S={t['sharpe']:>5} cagr={t['cagr']:>6} dd={t['maxdd']:<6} "
                      f"| va S={v['sharpe']:>5} cagr={v['cagr']:>6} | ciro/yil={t['ciro_yil']}")
    return out


def faz_gunluk(kaydet=True):
    """4. soru: DD dagilimi, ciro/yil, islem/yil (gelistirme = 2018-04..2025-12)"""
    out = {}
    for sp in ("s3_orij_bnb", "s3_2018_xrp", "yuruyen_top3", "s1_btc"):
        for kural, kw in (("ma50", {}), ("ma50_vt30", dict(hedef=0.30)), ("altut", dict(altut=True))):
            r = kos2(sepet_cfg(sp, **kw), ("gelistirme",), kaydet=kaydet, ad=f"{sp}__{kural}", not_="gunluk")
            g, gun, W = r["_g"], r["_gun"], r["_W"]
            ciro, giris = akis(W[:len(g)], gun[:len(g)])
            yl_c, yl_i = {}, {}
            for t, cc, ii in zip(gun[:len(g)], ciro, giris):
                y = datetime.fromtimestamp(t / 1e3, timezone.utc).year
                yl_c[y] = yl_c.get(y, 0.0) + float(cc); yl_i[y] = yl_i.get(y, 0.0) + float(ii)
            out[f"{sp}__{kural}"] = dict(
                gelistirme={k: v for k, v in r["gelistirme"].items() if not k.startswith("_")},
                en_kotu_5_dusus=dd_dagilim(g, gun, 5),
                ciro_yil={int(y): round(v, 2) for y, v in sorted(yl_c.items())},
                islem_yil={int(y): int(round(v)) for y, v in sorted(yl_i.items())})
            print(f"{sp}__{kural:<10} gelistirme S={out[f'{sp}__{kural}']['gelistirme'].get('sharpe')} "
                  f"dd={out[f'{sp}__{kural}']['gelistirme'].get('maxdd')} "
                  f"en_kotu5={[d['derinlik'] for d in out[f'{sp}__{kural}']['en_kotu_5_dusus']]}")
    return out


# ------------------------------------------------------------------ rapor metinleri
CEVAPLAR = {
    "S1_sepet_secim_yanliligi":
        "EVET, olculebilir bir yanlilik var - ama kuralin YONU sepetten bagimsiz. "
        "Ayni MA50 kurali 7 sepette: train Sharpe (ma50) 0.86-1.49, medyan 1.13; (vt30) 1.07-1.59, medyan 1.37. "
        "A1/A2'nin sepeti (BTC/ETH/BNB) HER IKI kuralda da 7 sepetin EN IYISI -> tepe noktasi secilmis. "
        "2018'de gercekten bilinebilir sepetler cok daha dusuk: {BTC,ETH,XRP} 1.13/1.37, "
        "{BTC,ETH,XRP,LTC,BCH} 0.86/1.07, YURUYEN aylik ilk-3 0.88/1.19. "
        "Yani train Sharpe'in ~0.2-0.6'si BNB'yi geriye donuk secmekten geliyor. "
        "Yuruyen sepet tarihcesi bunu dogruluyor: 2018-04..2020-01 BTC/ETH/LTC, 2020-10..2022-11 agirlikla "
        "BTC/ETH/YFI, BNB ilk kez 2022-12'de ilk-3'e giriyor. BNB 2018'de secilemezdi. "
        "BUNA RAGMEN: kural her sepette KENDI al-tutunu geciyor - train'de 7/7 (Sharpe farki ma50 +0.11..+0.43, "
        "vt30 +0.30..+0.54), valid'de ma50 6/7, vt30 7/7; maxDD her sepette al-tutun yarisi veya alti. "
        "SONUC: 'buyuk likit coinlerde MA50 + vol hedefleme' GENEL bir etki; ama 'Sharpe 1.5' SEPETE OZEL. "
        "Yeni bir sepet icin durust beklenti medyan degerlerdir: train ~1.15-1.40, valid ~0.7-0.8.",
    "S2_yil_bazinda":
        "8 yil (2018-2025). Getiri olarak strateji al-tutu sadece 3-5/8 yilda geciyor (sepete gore); "
        "maxDD olarak 6-8/8 yilda (vt30 icin 7 sepetin hepsinde 8/8) daha dusuk. "
        "Kazanc TAMAMEN ayi yillarindan: b3'te 2018 al-tut -49% vs ma50 -22% vs vt30 -12%; "
        "2022 al-tut -62% vs -35% vs -19%. Boga yillarinda geride kaliyor (2020: 317% vs 216% vs 88%; "
        "2023: 92% vs 79% vs 55%; 2024: 97% vs 56% vs 45%). Tek istisna 2025: al-tut +4% vs ma50 +44% vs vt30 +27%. "
        "Yani bu bir 'getiri arttirma' degil, 'dusus kesme' stratejisi; Sharpe ustunlugu paydadan geliyor.",
    "S3_uygulama":
        "(a) HAFTALIK (sadece pazartesi) kontrol NET ZARARLI: ciro 20.8 -> 8.9 (b3 ma50) ama Sharpe "
        "train -0.10..-0.24, valid -0.20..-0.35 dusuyor. 7bp maliyette tasarruf (~%0.8/yil) kaybedilen "
        "sinyal zamanlamasini karsilamiyor. Gunluk kontrol sart. "
        "(b) 1 GUN SINYAL GECIKMESI (t+2 acilista islem): train'de neredeyse etkisiz (+-0.05), ama VALID'de "
        "b3 icin agir: ma50 1.37->0.99, vt30 1.39->1.09. Yuruyen sepette daha az: 1.56->1.29 / 1.54->1.33. "
        "2025 performansinin kayda deger kismi sinyal sonrasi ilk gunun hareketinden geliyor -> gecikmeye duyarli. "
        "(c) MALIYET dayanikli: 7/10/20bp tek yon -> b3 vt30 train 1.59/1.57/1.51, valid 1.39/1.36/1.26. "
        "En kotu kombinasyon (1 gun gecikme + 20bp) bile valid'de ma50 0.89 / vt30 0.97, al-tutun (0.33) uzerinde.",
    "S4_sonuc_gunlugu":
        "A2 (b3 ma50 vt30, gelistirme 2018-04..2025-12, 2830 gun): en kotu 5 dusus %27.5 / %21.7 / %15.6 / "
        "%15.3 / %12.8. En kotusu 2021-11-07 tepe -> 2023-01-04 dip (423 gun dusus) + 308 gun toparlanma = "
        "731 gun (2 yil) su altinda. A1 (vt yok): %49.9 / %48.5 / %36.1 / %28.9 / %26.5, en kotusu 423+406=829 gun. "
        "Al-tut: %77.2 / %73.6 / %68.1 / %54.5 / %40.4. "
        "Ciro/yil: A2 13.9 (yillik 8.6-21.2), A1 21.0 -> 7bp'de yillik maliyet yuku %1.0 / %1.5; 20bp'de %2.8 / %4.2. "
        "Islem (pozisyon acilisi)/yil: her ikisi de ~31 (3 coin) = coin basina ~10-11 giris/yil; yillar arasi 26-34.",
}

DUSENLER = [
    "haftalik (pazartesi) rebalans: 6 konfigurasyonun hepsinde Sharpe dusuyor, ciro tasarrufu yetmiyor",
    "haftalik + 1 gun gecikme: en kotu uygulama kombinasyonu (b3 vt30 train 1.30)",
    "{BTC,ETH,LTC} sepeti: train 0.99/1.20, valid 0.18/0.32 - kural burada neredeyse hicbir sey uretmiyor",
    "{BTC,ETH,XRP,LTC,BCH} (2018 ilk-5): train 0.86/1.07, valid ma50 0.04 - al-tutun (0.28) ALTINDA",
    "tek {BTC}: valid'de vt30 0.13 (ma50 0.33); tek varlikta vol hedefleme 2025'te faydasiz",
]

OGRENILEN = [
    "Sepet secimi A1/A2'nin train Sharpe'inin ~0.2-0.6'sini aciklıyor: BTC/ETH/BNB, denenen 7 sepetin "
    "en iyisi ve 2018'de secilemezdi (yuruyen hacim sepetinde BNB ancak 2022-12'de ilk-3'e giriyor).",
    "Kuralin kendisi sepete duyarli DEGIL: 7 sepetin 7'sinde de kendi al-tutunu train Sharpe'te geciyor ve "
    "maxDD'yi yariliyor. 'Buyuk likit coinlerde MA50' genel; 'Sharpe 1.5' ozel. Beklentiyi medyana (train "
    "~1.15-1.40, valid ~0.7-0.8) cekmek gerekir.",
    "Ileriye bakmayan YURUYEN sepet (her ay basi 90g hacim ilk-3) uygulanabilir bir alternatif: train'de daha "
    "zayif (vt30 1.19) ama valid 2025'te en iyi (1.54) ve gecikmeye A2'den daha az duyarli. Sepet secimi "
    "sorununu yapisal olarak ortadan kaldiriyor - HOLDOUT'ta A2'nin yaninda kosulmali.",
    "Edge'in kaynagi getiri degil dusus kesme: 8 yilin sadece 3-5'inde al-tutu getiride geciyor, 6-8'inde "
    "DD'si dusuk. Bu, boga piyasasinda geride kalmayi kabul etmek demek - stratejinin gercek testi ayi yillari.",
    "Uygulama tarafinda tek kirilgan nokta SINYAL GECIKMESI: 1 gun gecikme valid Sharpe'i b3'te 1.39->1.09 "
    "dusuruyor. Maliyet (20bp'ye kadar) ve rebalans sikligi ikincil. Yani kapanis sonrasi ilk aciliste "
    "islem yapabilmek kritik; gunun ilerisine kaymak alfanin ~%25'ini yiyor.",
    "En kotu dusus 731 gun (A2) / 829 gun (A1) su altinda. Maksimum DD %27.5 kucuk gorunse de dayanma "
    "suresi 2 yil; canli kullanimda asil risk bu.",
]

ONERI = (
    "1) A2'yi (b3 MA50 + vt30) HOLDOUT'a goturmeye devam et AMA beklentiyi dusur: sepet secim yanliligi "
    "sonrasi durust tahmin Sharpe ~1.2-1.4 (train medyani), 2026'da 0.7-1.0 bile basari sayilir. "
    "2) YANINA yuruyen_top3 + vt30'u ekle (aylik, 90g USDT hacmi ilk-3, ileriye bakma yok). Bu, sepet "
    "secimini kuralin parcasi yapar; train 1.19 / valid 1.54. Iki adayin HOLDOUT'ta AYRISMASI bilgi verir: "
    "A2 yuruyen sepetten belirgin iyi cikarsa bu sansin devami, kotu cikarsa secim yanliliginin bedelidir. "
    "3) Gunluk kontrolu koru - haftalige gecme. 4) Sinyal gecikmesine duyarlilik nedeniyle canli uygulamada "
    "gunluk kapanis (00:00 UTC) sonrasi ilk saatlerde islem sart; bu mumkun degilse beklenen Sharpe'ten "
    "0.3 dus. 5) Yeni ajan icin: bu ailede parametre aramasi BITTI (tsmom 108 + tsmom2 150 konfigurasyon); "
    "kalan tek acik soru sinyal gecikmesinin intraday yapisi (1h veri ile: kapanistan sonra kac saat "
    "icinde girilirse alfa korunuyor)."
)

NOTLAR = [
    "Bu ailede TRAIN ve VALID sadece raporlandi; hicbir secim/optimizasyon yapilmadi. Yeni parametre "
    "aranmadi - tsmom.py'nin panel/oylar/volat/roll_mean fonksiyonlari import edilerek ayni MA50(+vt30) "
    "kurali farkli sepet ve uygulama varsayimlarinda kosuldu. HOLDOUT'a dokunulmadi.",
    "Tum konfigler ayni pencerede (2018-04-01 sonrasi, 2830 gun) ve harness varsayilan maliyetiyle (7bp) "
    "kosuldu; 10/20bp SADECE dayaniklilik testi olarak eklendi (maliyet dusurulmedi).",
    "Yuruyen sepet, o gunun kapanisinda bilinen 90g ortalama dolar hacmine (V*C) gore secilir; secim ayin "
    "ilk gununde bir ONCEKI gunun verisiyle yapilir -> ileriye bakma yok.",
    "SURVIVORSHIP: data1d_all bugun listede olan ~372 coin. Yuruyen sepet bu havuzdan seciyor; 2020-10..2022 "
    "arasi YFI secimi bu yanliligin bir belirtisi olabilir. Gercek zamanli sonuc muhtemelen biraz daha kotu.",
    "BCHUSDT verisi 2019-11-28'de basliyor; s5_2018 sepeti 2018-2019'da fiilen 4 coinle (BTC/ETH/XRP/LTC, "
    "XRP de 2018-05'ten itibaren) calisti. Maske bunu dogru kaldiriyor ama sepet o donemde eksik.",
]


# ------------------------------------------------------------------ rapor
def yaz(tablo, yil, uyg, gunluk, tarihce):
    d = [k for k in H.defter_oku() if k.get("aile") == AILE]
    n = len({(k["ad"], k["bolum"], json.dumps(k["params"], default=str, sort_keys=True)) for k in d})
    S = lambda t, e, b: t[e][b]["sharpe"]
    sepet_ozet = []
    for s in tablo:
        sepet_ozet.append(dict(
            ad=s["ad"], coinler=s["coinler"],
            ma50=dict(train=s["ma50"]["train"], valid=s["ma50"]["valid"]),
            ma50_vt30=dict(train=s["ma50_vt30"]["train"], valid=s["ma50_vt30"]["valid"]),
            altut=dict(train=s["altut"]["train"], valid=s["altut"]["valid"]),
            fark_sharpe=dict(train_ma50_eksi_altut=round(S(s, "ma50", "train") - S(s, "altut", "train"), 2),
                             valid_ma50_eksi_altut=round(S(s, "ma50", "valid") - S(s, "altut", "valid"), 2),
                             train_vt30_eksi_altut=round(S(s, "ma50_vt30", "train") - S(s, "altut", "train"), 2),
                             valid_vt30_eksi_altut=round(S(s, "ma50_vt30", "valid") - S(s, "altut", "valid"), 2))))
    tr = [S(s, "ma50_vt30", "train") for s in tablo]; va = [S(s, "ma50_vt30", "valid") for s in tablo]
    tr1 = [S(s, "ma50", "train") for s in tablo]; va1 = [S(s, "ma50", "valid") for s in tablo]
    out = dict(
        aile=AILE, denenen=n, kaynak="1d", bolum="train<2025 | valid 2025 (ikisi de sadece RAPOR)",
        analiz_baslangici="2018-04-01 (tsmom ile ayni pencere)",
        amac="A1/A2'nin sepet secim yanliligi + uygulama dayanikliligi. Yeni parametre aramasi yok.",
        sepetler=sepet_ozet,
        yuruyen_sepet_tarihcesi=tarihce,
        ozet_istatistik=dict(
            ma50_train_sharpe=dict(min=round(min(tr1), 2), max=round(max(tr1), 2), medyan=round(float(np.median(tr1)), 2)),
            ma50_valid_sharpe=dict(min=round(min(va1), 2), max=round(max(va1), 2), medyan=round(float(np.median(va1)), 2)),
            vt30_train_sharpe=dict(min=round(min(tr), 2), max=round(max(tr), 2), medyan=round(float(np.median(tr)), 2)),
            vt30_valid_sharpe=dict(min=round(min(va), 2), max=round(max(va), 2), medyan=round(float(np.median(va)), 2))),
        yil_bazinda=yil, uygulama=uyg, sonuc_gunlugu=gunluk,
        cevaplar=CEVAPLAR, dusenler=DUSENLER, ogrenilen=OGRENILEN, oneri=ONERI, notlar=NOTLAR,
    )
    p = os.path.join(LAB, "sonuc_tsmom2.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("yazildi:", p, "| denenen:", n)
    return out, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="hepsi", choices=["hepsi", "sepet", "yil", "uygulama", "gunluk", "tarihce"])
    a = ap.parse_args()
    P = T.panel()
    print(f"panel: {len(P['syms'])} sembol, {len(P['gun'])} gun, bas idx={P['bas']} "
          f"({datetime.fromtimestamp(P['gun'][P['bas']]/1e3, timezone.utc).date()})")
    if a.faz == "tarihce":
        for d, L in sepet_tarihce():
            print(d, L)
        return
    if a.faz in ("hepsi", "sepet"):
        print("\n=== 1) SEPET DUYARLILIGI ===")
        tablo = faz_sepet()
    if a.faz in ("hepsi", "yil"):
        print("\n=== 2) YIL BAZINDA ===")
        yil = faz_yil()
    if a.faz in ("hepsi", "uygulama"):
        print("\n=== 3) UYGULAMA ===")
        uyg = faz_uygulama()
    if a.faz in ("hepsi", "gunluk"):
        print("\n=== 4) SONUC GUNLUGU ===")
        gunluk = faz_gunluk()
    if a.faz == "hepsi":
        th = sepet_tarihce()
        print("\n=== yuruyen sepet bilesim degisimleri:", len(th))
        for d, L in th[:12]:
            print("  ", d, L)
        yaz(tablo, yil, uyg, gunluk, th)


if __name__ == "__main__":
    main()
