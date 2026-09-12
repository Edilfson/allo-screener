"""kesif_tavan - KESIF ailesi: tek-coin agirlik TAVANI ve SEPET GENISLIGI, gap riskine karsi.

BU AILE HOLDOUT ICIN DEGIL. Amac: canli kagit portfoye 3. varyant olarak eklenip 12 ay
izlenecek H5 hipotezini secmek. Sadece 'gelistirme' (train+valid, <2026-01-01) bolumu
raporlanir; HOLDOUT'a dokunulmaz, LAB_HOLDOUT ayarlanmaz.

Cikis noktasi: kirmizi takim bulgu 3A -> Y (yuruyen top-3 + MA50 + vt30) sepetinde
tek-coin agirligi ort ~0.19 (p95 0.37); MA50 trend filtresi bir GECELIK -%90 cokuse
karsi HICBIR koruma vermez, vol hedefleme de yalniz 30 gunluk GECMIS vole bakar.

Sorular:
  1) w_i <= {0.10, 0.15, 0.20, 0.25} tavani (fazlasi NAKDE, yeniden dagitim YOK)
     A2 (BTC/ETH/BNB) ve Y (yuruyen top-3) Sharpe/maxDD'sini ne kadar degistiriyor?
  2) Sepet genisligi: yuruyen top-3 -> top-4/5/8 (esit agirlik tabani ayni) ne yapiyor?
  3) GAP STRES TESTI: gelistirme doneminde HER YIL rastgele bir gun, o gun TUTULAN
     rastgele bir coine -%50 / -%90 gecelik sok. 100 tekrar. Equity kaybi dagilimi,
     tavanli vs tavansiz.

Yontem:
  - tsmom2.agirliklar2(cfg) ciktisi alinir, W = min(W, tavan) uygulanir (long-only, W>=0).
  - Olcum kos2 ile ayni: H.sim_portfoy + H.portfoy_rapor.
  - Sok, O matrisinin KOPYASINDA fiyat kaydirmasiyla uygulanir: O2[t+2:, j] *= (1+s).
    Boylece sok gunu getirisi tam s olur, sonraki gunlerin GORECELI getirileri degismez
    -> olculen sey yalnizca gap'in kendisi (temiz, kompoze edilebilir, ileriye bakma yok).
  - Ayni (gun, coin) sok cizelgesi tum tavan varyantlarinda AYNI (eslesmis karsilastirma).

Kullanim:
  PYTHONIOENCODING=utf-8 python kesif_tavan.py            # hepsi -> sonuc_kesif_tavan.json
  PYTHONIOENCODING=utf-8 python kesif_tavan.py --faz grid
  PYTHONIOENCODING=utf-8 python kesif_tavan.py --faz stres --tekrar 100
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import tsmom as T
import tsmom2 as T2

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "kesif_tavan"
BOLUM = "gelistirme"
MALIYET = T2.MALIYET_VARSAYILAN
TAVANLAR = [None, 0.25, 0.20, 0.15, 0.10]
TOHUM = 20260913

# A2 ve Y'nin tanimlari (tsmom2 ile birebir)
def cfg_a2():
    return T2.sepet_cfg("s3_orij_bnb", hedef=0.30)


def cfg_y(top=3):
    return T2.C2(f"yuruyen_top{top}", evren="aylik_top", top=top, hedef=0.30)


# ------------------------------------------------------------------ tavan
def tavan_uygula(W, tavan):
    """w_i <= tavan; fazlasi NAKDE kalir (yeniden dagitim yok). W long-only (>=0)."""
    if tavan is None:
        return W
    return np.minimum(W, tavan)


_WC = {}
def W_of(cfg, tavan):
    k = (cfg["ad"], cfg.get("top"), tavan)
    if k not in _WC:
        _WC[k] = tavan_uygula(T2.agirliklar2(cfg), tavan)
    return _WC[k]


def agirlik_istat(Wb):
    """tutulan pozisyonlarin agirlik dagilimi + tavanin ne kadar isirdigi"""
    nz = Wb[Wb > 1e-9]
    if len(nz) == 0:
        return {}
    return dict(w_ort=round(float(nz.mean()), 4), w_p95=round(float(np.percentile(nz, 95)), 4),
                w_max=round(float(nz.max()), 4), poz_gun=round(float((Wb > 1e-9).sum(1).mean()), 2))


# ------------------------------------------------------------------ kosu
def kos_tavan(cfg, tavan, ad=None, kaydet=True, not_="", O=None, olcek=1.0):
    P = T.panel(); b = P["bas"]
    W = W_of(cfg, tavan)
    if olcek != 1.0:
        W = W * olcek
    Ok = P["O"] if O is None else O
    eq, g = H.sim_portfoy(Ok[b:], P["C"][b:], W[b:], maliyet=MALIYET)
    gun = P["gun"][b:]
    r = H.portfoy_rapor(eq, g, gun, BOLUM)
    ciro, giris = T2.akis(W[b:], gun)
    m = np.ones(len(g), bool)          # gelistirme = tum kesilmis veri (holdout fiziksel kesik)
    a, bb = H.BOLUM[BOLUM]
    m = (gun[:len(g)] >= a) & (gun[:len(g)] < bb)
    yil = max(m.sum() / 365, 1e-9)
    r["ciro_yil"] = round(float(ciro[:len(g)][m].sum() / yil), 2)
    r["ort_gross"] = round(float(np.abs(W[b:][:len(g)][m]).sum(1).mean()), 3)
    r.update(agirlik_istat(W[b:][:len(g)][m]))
    r["tavan"] = tavan
    r["olcek"] = round(float(olcek), 3)
    ad = ad or f"{cfg['ad']}__tavan{'yok' if tavan is None else int(tavan*100)}"
    if kaydet:
        H.kaydet(AILE, ad, dict(cfg, tavan=tavan, maliyet=MALIYET), BOLUM, r, not_)
    return ad, r, (eq, g, gun)


def faz_grid(kaydet=True):
    """1+2. soru: A2 / Y3 / Y4 / Y5 / Y8 x tavan {yok,25,20,15,10}"""
    aileler = [("A2_b3", cfg_a2())] + [(f"Y{n}_yuruyen{n}", cfg_y(n)) for n in (3, 4, 5, 8)]
    tablo = {}
    for etiket, cfg in aileler:
        g0 = None
        for tv in TAVANLAR:
            ad = f"{etiket}__tavan{'yok' if tv is None else int(tv*100)}"
            _, r, _ = kos_tavan(cfg, tv, ad=ad, kaydet=kaydet,
                                not_="kesif; holdout icin degil; H5 adayi taramasi")
            tablo[ad] = r
            print(f"  {ad:28s} S={r['sharpe']:5.2f} dd={r['maxdd']:.3f} cagr={r['cagr']:6.3f} "
                  f"gross={r['ort_gross']:.3f} wmax={r.get('w_max', 0):.3f} ciro={r['ciro_yil']:.1f}")
            if tv is None:
                g0 = r["ort_gross"]
                continue
            # KIRMIZI TAKIM 6B: maxDD kazanci 'daha az yatirim yapmaktan' mi geliyor?
            # Tavanli portfoyu AYNI ortalama gross'a olcekleyip tekrar bak (tanisal; tavani nominal olarak asar).
            k = g0 / max(r["ort_gross"], 1e-9)
            ad2 = ad + "__grossesit"
            _, r2, _ = kos_tavan(cfg, tv, ad=ad2, kaydet=kaydet, olcek=k,
                                 not_="tanisal: olcek etkisini ayirmak icin gross-esitlenmis tavanli portfoy")
            tablo[ad2] = r2
            print(f"    {'^ gross-esit (x%.2f)' % k:26s} S={r2['sharpe']:5.2f} dd={r2['maxdd']:.3f} "
                  f"cagr={r2['cagr']:6.3f} gross={r2['ort_gross']:.3f}")
    return tablo


# ------------------------------------------------------------------ gap stres
def sok_cizelge(Wb, gun, rng, yillar=None):
    """Her yil icin: (t, j) - o gun TUTULAN coinlerden rastgele biri."""
    yl = np.array([datetime.fromtimestamp(x / 1e3, timezone.utc).year for x in gun])
    out = []
    for y in (yillar or sorted(set(yl.tolist()))):
        idx = np.flatnonzero((yl == y) & (np.arange(len(yl)) < len(Wb) - 2))
        idx = [t for t in idx if (Wb[t] > 1e-9).any()]
        if not idx:
            continue
        t = int(rng.choice(idx))
        tut = np.flatnonzero(Wb[t] > 1e-9)
        j = int(rng.choice(tut))
        out.append((t, j, y))
    return out


def sokla(O, b, soklar, s):
    """O kopyasinda kalici fiyat kaydirmasi: sok gunu getirisi = s, sonraki goreli getiriler ayni."""
    O2 = O.copy()
    for (t, j, _y) in soklar:
        O2[b + t + 2:, j] *= (1.0 + s)
    return O2


def faz_stres(tekrar=100, soklar_pct=(-0.50, -0.90), kaydet=True, aileler=None, olcek_hedef=None, etiket_ek=""):
    """olcek_hedef: verilirse her konfig bu ortalama gross'a olceklenir (ESIT RISK butcesi karsilastirmasi;
    tavanin 'daha az yatirim yapma' etkisini ayirir - kirmizi takim 6B kurali)."""
    P = T.panel(); b = P["bas"]; gun = P["gun"][b:]
    aileler = aileler or [("A2_b3", cfg_a2()), ("Y3_yuruyen3", cfg_y(3)), ("Y5_yuruyen5", cfg_y(5))]
    cikti = {}
    for etiket, cfg in aileler:
        Wref = W_of(cfg, None)[b:]                       # sok cizelgesi TAVANSIZ holdinglerden
        rng = np.random.default_rng(TOHUM)
        cizelge = [sok_cizelge(Wref, gun, rng) for _ in range(tekrar)]
        for s in soklar_pct:
            for tv in TAVANLAR:
                W = W_of(cfg, tv)
                if olcek_hedef:
                    W = W * (olcek_hedef / max(float(np.abs(W[b:]).sum(1).mean()), 1e-9))
                eq0, g0 = H.sim_portfoy(P["O"][b:], P["C"][b:], W[b:], maliyet=MALIYET)
                taban = H.portfoy_rapor(eq0, g0, gun, BOLUM)
                kayip, gun_kayip, dds, sharpes = [], [], [], []
                for cz in cizelge:
                    O2 = sokla(P["O"], b, cz, s)
                    eq, g = H.sim_portfoy(O2[b:], P["C"][b:], W[b:], maliyet=MALIYET)
                    r = H.portfoy_rapor(eq, g, gun, BOLUM)
                    kayip.append(1.0 - (1 + r["toplam"]) / (1 + taban["toplam"]))
                    dds.append(r["maxdd"]); sharpes.append(r["sharpe"])
                    # tek gunluk en kotu darbe (o gun tutulan agirlik * sok)
                    gun_kayip.append(max(abs(float(W[b:][t, j]) * s) for (t, j, _y) in cz))
                k = np.array(kayip); d = np.array(dds); sh = np.array(sharpes)
                gk = np.array(gun_kayip)
                ad = f"{etiket}__sok{int(abs(s)*100)}__tavan{'yok' if tv is None else int(tv*100)}{etiket_ek}"
                r = dict(tavan=tv, sok=s, tekrar=tekrar, olcek_hedef=olcek_hedef,
                         gross=round(float(np.abs(W[b:]).sum(1).mean()), 3),
                         taban_sharpe=taban["sharpe"], taban_maxdd=taban["maxdd"],
                         taban_toplam=taban["toplam"],
                         kayip_ort=round(float(k.mean()), 4), kayip_med=round(float(np.median(k)), 4),
                         kayip_p95=round(float(np.percentile(k, 95)), 4),
                         kayip_max=round(float(k.max()), 4),
                         sharpe_ort=round(float(sh.mean()), 3), sharpe_p05=round(float(np.percentile(sh, 5)), 3),
                         maxdd_ort=round(float(d.mean()), 3), maxdd_p95=round(float(np.percentile(d, 95)), 3),
                         tekgun_ort=round(float(gk.mean()), 4), tekgun_p95=round(float(np.percentile(gk, 95)), 4),
                         tekgun_max=round(float(gk.max()), 4))
                cikti[ad] = r
                if kaydet:
                    H.kaydet(AILE, ad, dict(cfg, tavan=tv, sok=s, tekrar=tekrar, olcek_hedef=olcek_hedef,
                                            maliyet=MALIYET), BOLUM, r,
                             "gap stres testi (yil basina 1 sok, 8 yil, %d tekrar)" % tekrar)
                print(f"  {ad:36s} kayip ort={r['kayip_ort']:.3f} p95={r['kayip_p95']:.3f} "
                      f"maxdd {taban['maxdd']:.3f}->{r['maxdd_ort']:.3f} S {taban['sharpe']:.2f}->{r['sharpe_ort']:.2f} "
                      f"tekgun p95={r['tekgun_p95']:.3f}")
    return cikti


# ------------------------------------------------------------------ rapor
def faz_esitrisk(tekrar=100, kaydet=True):
    """ESIT RISK BUTCESI: her konfig Y3-tavansiz'in ortalama gross'una olceklenir.
    Soru: tavan/genislik, AYNI kadar yatirim yapildiginda gap riskini gercekten dusuruyor mu?"""
    P = T.panel(); b = P["bas"]
    hedef = float(np.abs(W_of(cfg_y(3), None)[b:]).sum(1).mean())
    print(f"  hedef ortalama gross = {hedef:.3f}")
    return hedef, faz_stres(tekrar, (-0.90,), kaydet,
                            aileler=[("A2_b3", cfg_a2()), ("Y3_yuruyen3", cfg_y(3)), ("Y5_yuruyen5", cfg_y(5))],
                            olcek_hedef=hedef, etiket_ek="__esitrisk")


OGRENILEN = [
    "TAVAN 'BEDAVA MAXDD' DEGIL, OLCEK KUCULTMEDIR. Y3'te tavan 0.10 maxDD'yi 0.273->0.172 "
    "dusuruyor gibi gorunuyor ama ortalama gross da 0.244->0.141 dusuyor. AYNI gross'a "
    "olceklenince maxDD 0.273->0.285 (DAHA KOTU) ve Sharpe 1.37->1.19. A2'de ayni: "
    "gross-esit tavan0.15 maxDD 0.275->0.304, Sharpe 1.56->1.46. Kirmizi takim 6B kurali "
    "tavan testinde de aynen gecerli.",
    "SABIT RISK BUTCESINDE TAVAN BEKLENEN GAP HASARINI DEGISTIRMEZ, SADECE KUYRUGU KESER. "
    "Olay basina beklenen darbe = gross * |sok| / K_ort (K = ort. tutulan coin sayisi); tavan "
    "gross'u ve K'yi degistirmez, yalniz agirlik dagilimini duzlestirir. Olcum bunu birebir "
    "dogruluyor: gross-esit -90% sok, 8 yil x 8 olay kumulatif kayip Y3 tavansiz 0.732 vs "
    "tavan0.10 0.725 (fark yok); ama tek gunluk en kotu darbe p95 0.371 -> 0.155 (-58%).",
    "GAP RISKINI GERCEKTEN DUSUREN SEY SEPET GENISLIGI. Esit gross'ta (0.244) kumulatif -90% "
    "sok kaybi: top-3 0.732, top-5 0.558 (-24% goreli); tek gunluk darbe p95 0.371 -> 0.259. "
    "Cunku K 1.46'dan 2.40'a cikiyor ve beklenen darbe 1/K ile azaliyor - tavanin yapamadigi sey.",
    "GENISLIK BEDAVA DEGIL VE YON SON 2 YILDA TERS. Gelistirme genelinde top-5 Sharpe 1.40 / "
    "maxDD 0.236, top-3 1.37 / 0.273, top-4 1.37 / 0.246, top-8 1.22 / 0.229 -> 3-5 arasi duz "
    "bir bolge, 8'de bozuluyor. Ama yil bazinda 2024/2025 getirisi top3 0.36/0.30, top4 0.40/0.20, "
    "top5 0.31/0.15, top8 0.25/0.06: genislik son iki yilda MONOTON zarar ettiriyor. "
    "Yani top-5 ustunlugu 2018-2023'ten geliyor; ileriye donuk kanit degil, izlenmesi gereken hipotez.",
    "TAVANIN ISIRMA NOKTASI SEPETE BAGLI. 3 coinli sepette dogal tek-coin agirligi ort 0.17 / "
    "p95 0.31 / max 0.47 -> 0.25 tavani bile isiriyor (Sharpe -0.07). 5 coinli sepette ort 0.09 / "
    "p95 0.17 / max 0.30 -> 0.25 tavani hic isirmiyor (Sharpe -0.00), 0.20 tavani sadece "
    "-0.02 Sharpe / +0.001 maxDD'ye mal oluyor. Yani tavan, GENIS sepette neredeyse bedava bir emniyet raylidir.",
    "TAVAN NAKIT ORANINI BUYUTUR VE MODEL NAKDE %0 FAIZ YAZAR. Y3 tavan0.10'da ortalama gross "
    "0.141 - portfoyun %86'si atil. Canlida bu nakde faiz/stablecoin getirisi islenirse tavanli "
    "varyantin Sharpe'i buradaki olcumden YUKSEK cikar; yani tavan aleyhine olcum muhafazakardir.",
]

ONERI = {
    "H5_karar": "Tavani TEK BASINA koyma. H5 = 'yuruyen top-5 esit agirlik + MA50 + vt30 + tek-coin tavani 0.20' "
                "(fazlasi nakde, yeniden dagitim yok). Yani asil mudahale SEPET GENISLIGI (3->5); tavan onun "
                "uzerine ucuz bir kuyruk emniyeti olarak eklenir.",
    "neden_tavan_tek_basina_HAYIR": "Sabit risk butcesinde tavan beklenen gap hasarini degistirmiyor "
        "(gross-esit -90% sok kumulatif kayip Y3: 0.732 tavansiz vs 0.725 tavan0.10) ve maxDD'yi "
        "OLCEK ETKISI ayiklandiginda kotulestiriyor (Y3 0.273->0.285, A2 0.275->0.304). Ham tabloda "
        "gorunen maxDD iyilesmesi tamamen 'daha az yatirim yapmak'; ayni sonucu tum portfoyu 0.6 ile "
        "carparak bedavaya elde edersin. Ayrica dar sepette pahali: Y3'te tavan 0.10 Sharpe'i -0.18 dusuruyor.",
    "neden_genislik_EVET": "Beklenen olay basina darbe = gross*|sok|/K. top-3 -> top-5, K 1.46->2.40; "
        "esit gross'ta kumulatif -90% sok kaybi 0.732 -> 0.558, tek gunluk en kotu darbe p95 0.371 -> 0.259. "
        "Bunu tavan yapamaz. Ustelik gelistirme doneminde Sharpe DUSMUYOR (1.37 -> 1.40) ve maxDD dusuyor (0.273 -> 0.236).",
    "neden_tavan_0.20_(0.10_degil)": "5 coinli sepette dogal p95 zaten 0.17; 0.25 tavani hic isirmiyor, "
        "0.20 tavani sadece pahali gunleri kirpiyor: Sharpe 1.40->1.38, maxDD 0.236->0.237, CAGR 0.271->0.264, "
        "buna karsilik -90% gap'te en kotu tek gunluk darbe 0.255 -> 0.180 (kesin ust sinir 0.20). "
        "0.15 tavani ek koruma veriyor (darbe 0.135) ama Sharpe 1.35'e iniyor (-0.05); 0.10 gereksiz pahali. "
        "0.20 tavaninin asil degeri ORTALAMA degil GARANTI:'tek coin portfoyun %20'sinden fazlasi olamaz' "
        "canlida uygulanabilir, denetlenebilir ve operatorun paniklemesini onleyen sert bir kural.",
    "beklenen_fark_H5_vs_Y": {
        "temel": "Y = yuruyen top-3 + MA50 + vt30 (tavansiz), gelistirme (2018-04..2025-12, 2830 gun)",
        "sharpe": "1.37 -> 1.38  (+0.01; fark yok saymak dogru - genislik secimi Sharpe icin YAPILMIYOR)",
        "maxdd": "0.273 -> 0.237  (-3.6 puan)",
        "cagr": "0.292 -> 0.264  (-2.8 puan; ortalama gross 0.244 -> 0.219, yani daha az yatirim)",
        "ciro_yil": "13.6 -> 12.0",
        "tek_coin_agirlik": "ort 0.167 -> 0.091 | p95 0.307 -> 0.170 | max 0.472 -> 0.200 (sert tavan)",
        "gap_-90pct_tek_gun_darbe": "ort 0.264 -> 0.152 | p95 0.371 -> 0.180 | max 0.424 -> 0.180",
        "gap_-50pct_tek_gun_darbe": "p95 0.206 -> 0.100",
        "gap_8yil_8olay_kumulatif_kayip_(-90%)": "0.732 -> 0.513",
        "gap_sonrasi_sharpe_(-90%)": "0.47 -> 0.79 | gap sonrasi maxDD 0.462 -> 0.319",
    },
    "risk_ve_uyari": [
        "TERS KANIT: sepet genisligi son iki yilda ZARAR ettiriyor (2025 getirisi top3 +0.30, top4 +0.20, "
        "top5 +0.15, top8 +0.06; 2024 icin 0.36/0.40/0.31/0.25). Top-5'in ustunlugu 2018-2023 agirlikli. "
        "H5'i 'daha iyi strateji' diye degil, 'ayni Sharpe'i daha az gap riskiyle aliyor muyum' sorusunu "
        "12 ay ileriye donuk test etmek icin canliya koy.",
        "Bu aile KESIF: 20 grid + 15 gross-esit tanisal + 45 stres konfigurasyonu kosuldu, hepsi deftere "
        "yazildi. Coklu test yuku yuksek; buradan HOLDOUT'a aday CIKMAZ, holdout'a dokunulmadi.",
        "Stres testi tasarim geregi 'tavan beklenen hasari dusurmez' sonucunu bir olcude ICERIR "
        "(sok gorulen coin, tutulan coinler arasindan DUZGUN dagilimla seciliyor). Gercek hayatta cokme "
        "olasiligi buyuk/likit coinlerde daha dusuk olabilir - o durumda tavan daha da az degerli olur, "
        "cunku tavan tam da en buyuk (= en likit, en guvenli) pozisyonu kirpar. Bu, tavan aleyhine ek bir argumandir.",
        "Sok modeli kalici fiyat kaydirmasi: sok gunu getirisi tam s, sonraki GORECELI getiriler degismez. "
        "Yani stratejinin soka TEPKISI (MA50 kirilinca cikmasi) modellenmiyor; olculen sey yalnizca "
        "gap'in kendisi. Bu dogru olcum, cunku bir gecelik gap'e karsi cikis kurali zaten koruma saglamaz.",
        "Nakde %0 faiz varsayimi tavanli/genis varyantlar aleyhine calisiyor (gross 0.22-0.24, portfoyun "
        "~%77'si atil). Canli kagit portfoyde nakit getirisi de kaydedilmeli.",
    ],
    "canli_izleme_plani": [
        "H5'i 3. varyant olarak, A2 ve Y ile AYNI sermaye ve AYNI maliyet varsayimiyla kagit uzerinde kos.",
        "Aylik kaydet: gerceklesen Sharpe, maxDD, ortalama gross, ortalama/max tek-coin agirligi, "
        "tavanin kac gun isirdigi ve nakde birakilan agirlik.",
        "Onceden yazilan basari olcutu (12 ay sonunda): H5'in maxDD'si Y'ninkinden dusuk VE Sharpe farki "
        "-0.20'den kotu degil ise genislik+tavan kalici hale getirilir. Tek bir gercek gap olayi (-%40 uzeri "
        "gecelik dusus) yasanirsa, o olaydaki gozlenen darbe farki tek basina belirleyici kanit sayilir.",
    ],
}


def yaz(grid, stres, esitrisk=None):
    def g(a):
        return grid.get(a, {})

    def fark(a, b, alan):
        x, y = g(a).get(alan), g(b).get(alan)
        return None if x is None or y is None else round(y - x, 3)

    ozet = {}
    for etiket in ("A2_b3", "Y3_yuruyen3", "Y4_yuruyen4", "Y5_yuruyen5", "Y8_yuruyen8"):
        yok = f"{etiket}__tavanyok"
        if yok not in grid:
            continue
        ozet[etiket] = {
            "tavansiz": {k: g(yok).get(k) for k in ("sharpe", "maxdd", "cagr", "ort_gross", "w_ort", "w_p95", "w_max", "ciro_yil")},
            "tavan_farki": {f"t{int(tv*100)}": {
                "d_sharpe": fark(yok, f"{etiket}__tavan{int(tv*100)}", "sharpe"),
                "d_maxdd": fark(yok, f"{etiket}__tavan{int(tv*100)}", "maxdd"),
                "d_cagr": fark(yok, f"{etiket}__tavan{int(tv*100)}", "cagr"),
                "gross": g(f"{etiket}__tavan{int(tv*100)}").get("ort_gross"),
                "w_p95": g(f"{etiket}__tavan{int(tv*100)}").get("w_p95"),
            } for tv in TAVANLAR if tv is not None},
        }
    out = dict(aile=AILE, tarih=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
               bolum=BOLUM, holdout="DOKUNULMADI (LAB_HOLDOUT ayarlanmadi)",
               amac="H5: canli kagit portfoye 3. varyant - tek-coin tavani ve/veya sepet genisligi",
               denenen_sayisi=len(grid) + len(stres) + len((esitrisk or {}).get("tablo", {})),
               grid=grid, ozet=ozet, stres=stres, esitrisk=esitrisk or {},
               ogrenilen=OGRENILEN, oneri=ONERI)
    p = os.path.join(LAB, "sonuc_kesif_tavan.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("yazildi:", p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="hepsi", choices=["hepsi", "grid", "stres", "esitrisk"])
    ap.add_argument("--tekrar", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    grid, stres, esit = {}, {}, {}
    if a.faz in ("hepsi", "grid"):
        print("== grid (tavan x sepet genisligi), bolum=gelistirme")
        grid = faz_grid()
    if a.faz in ("hepsi", "stres"):
        print(f"== gap stres testi ({a.tekrar} tekrar x 8 yil)")
        stres = faz_stres(a.tekrar)
    if a.faz in ("hepsi", "esitrisk"):
        print("== esit risk butcesi (gross esitlenmis) gap stresi, sok -90%")
        hedef, tab = faz_esitrisk(a.tekrar)
        esit = dict(hedef_gross=round(hedef, 3), tablo=tab)
    if a.faz == "hepsi":
        yaz(grid, stres, esit)
    print("sure", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
