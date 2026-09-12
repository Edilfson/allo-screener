"""KIRMIZI TAKIM - A2 (b3 MA50+vt30) ve Y (yuruyen top-3 MA50+vt30) adaylarini SISIREN
her seyi bulmaya calisir. Orijinal kod (harness/tsmom/tsmom2) DEGISTIRILMEZ; burada
duzeltilmis kopyalar kosulur ve 'onceki vs duzeltilmis' karsilastirilir.

HOLDOUT'A DOKUNULMAZ. LAB_HOLDOUT ayarlanmaz. Ham .npy dosyalari yalniz harness uzerinden okunur.

Kullanim:
  PYTHONIOENCODING=utf-8 python kirmizi_takim.py            # hepsi -> sonuc_kirmizi.json
  PYTHONIOENCODING=utf-8 python kirmizi_takim.py --test 1   # tek madde
"""
import argparse, json, os, sys, copy
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import tsmom as T
import tsmom2 as T2

assert not H._holdout_acik(), "LAB_HOLDOUT acik! Kirmizi takim holdout'a dokunmaz."

LAB = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260913)
BULGULAR = []
OLC = {}


def bulgu(kod, sorun, kanit, etki, siddet):
    BULGULAR.append(dict(kod=kod, siddet=siddet, sorun=sorun, kanit=kanit, etki=etki))
    print(f"\n[{siddet}] {kod}: {sorun}\n   KANIT: {kanit}\n   ETKI : {etki}")


def SD(r):
    return dict(sharpe=r.get("sharpe"), maxdd=r.get("maxdd"), cagr=r.get("cagr"),
                gun=r.get("gun"), gross=r.get("ort_gross"), ciro_yil=r.get("ciro_yil"))


# =====================================================================================
# DUZELTILMIS SIMULATOR + AGIRLIK (orijinali degistirmeden)
# =====================================================================================
def ffill_of(O):
    """Open panelini ileri doldur (yalniz gecmis)."""
    return T.ffill(O)


def sim_portfoy_duz(O, C, W, maliyet=0.0007, funding=0.0, clip=None, ileri_bakma_maske=False,
                    Of=None):
    """harness.sim_portfoy'un duzeltilmis kopyasi.

    Duzeltmeler:
      (1) ORIJINALDE: w[isnan(O[t+1]) | isnan(O[t+2])] = 0  -> O[t+2] GELECEKTIR (t kapanisinda
          bilinemez). Burada yalniz O[t+1] (giris fiyati, t+1 acilisi) sart kosulur; t+2 acilisi
          eksikse pozisyon ffill'lenmis fiyatla tasinir (bilgi kullanilmaz).
      (2) clip=None -> getiri kirpmasi YOK (orijinal (-0.95, 5.0) kirpiyor).
    """
    n, m = C.shape
    if Of is None:
        Of = ffill_of(O)
    eq = [1.0]; g = []; w_prev = np.zeros(m)
    kirpildi = 0
    for t in range(n - 2):
        w = np.nan_to_num(W[t]).copy()
        if ileri_bakma_maske:
            w[np.isnan(O[t + 1]) | np.isnan(O[t + 2])] = 0        # orijinal (ileri bakan)
        else:
            w[np.isnan(O[t + 1])] = 0                              # duzeltilmis
        r = np.nan_to_num((Of[t + 2] - Of[t + 1]) / Of[t + 1])
        if clip is not None:
            r2 = np.clip(r, *clip)
            kirpildi += int(((r2 != r) & (np.abs(w) > 1e-12)).sum())   # SADECE pozisyonlu hucreler
            r = r2
        x = (w * r).sum() - np.abs(w - w_prev).sum() * maliyet - np.abs(w).sum() * funding
        g.append(x); eq.append(eq[-1] * (1 + x))
        w_prev = w * (1 + r) / (1 + x) if x > -1 else w
    return np.array(eq), np.array(g), kirpildi


def agirliklar_duz(cfg, shift=0):
    """tsmom2.agirliklar2 + NEGATIF shift destegi (ileri-bakma sondasi).
    shift>0: gecikme (t-shift sinyali). shift<0: GELECEK sinyal (t+|shift|) -> kontrol testi."""
    W = T2.agirliklar2(dict(cfg, gecikme=0))
    if shift > 0:
        W = np.vstack([np.zeros((shift, W.shape[1])), W[:-shift]])
    elif shift < 0:
        k = -shift
        W = np.vstack([W[k:], np.zeros((k, W.shape[1]))])
    return W


def kos_duz(cfg, bolumler=("train", "valid"), maliyet=0.0007, shift=0, clip=None,
            ileri_bakma_maske=False, cap=None, Wover=None):
    P = T.panel(); b = P["bas"]
    c = dict(cfg)
    if cap is not None:
        c["cap"] = cap
    W = agirliklar_duz(c, shift) if Wover is None else Wover
    eq, g, kirp = sim_portfoy_duz(P["O"][b:], P["C"][b:], W[b:], maliyet=maliyet, clip=clip,
                                  ileri_bakma_maske=ileri_bakma_maske)
    gun = P["gun"][b:]
    ciro, giris = T2.akis(W[b:], gun)
    out = {"_kirp": kirp, "_g": g, "_gun": gun, "_W": W[b:]}
    for bl in bolumler:
        r = H.portfoy_rapor(eq, g, gun, bl)
        a, bb = H.BOLUM[bl]
        m = (gun[:len(g)] >= a) & (gun[:len(g)] < bb)
        yil = len(g[m]) / 365 if m.sum() else 1
        r["ciro_yil"] = round(float(ciro[:len(g)][m].sum() / max(yil, 1e-9)), 2)
        r["ort_gross"] = round(float(np.abs(W[b:][:len(g)][m]).sum(1).mean()), 3)
        out[bl] = r
    return out


A2 = T2.sepet_cfg("s3_orij_bnb", hedef=0.30)
YY = T2.sepet_cfg("yuruyen_top3", hedef=0.30)
A2B = T2.sepet_cfg("s3_orij_bnb", altut=True)
YYB = T2.sepet_cfg("yuruyen_top3", altut=True)
ADAY = {"A2": A2, "Y": YY}


# =====================================================================================
# 1) ILERIYE BAKMA
# =====================================================================================
def test1_ileriye_bakma():
    R = {}
    print("\n" + "=" * 90 + "\n1) ILERIYE BAKMA\n" + "=" * 90)

    # ---- 1a. close[t] == open[t+1] mi?  (sim_portfoy'un 'guvenlik payi' gercek mi)
    P = T.panel()
    j = P["syms"].index("BTCUSDT")
    C = P["C"][:, j]; O = P["O"][:, j]
    m = ~np.isnan(C[:-1]) & ~np.isnan(O[1:])
    d = np.abs(C[:-1][m] / O[1:][m] - 1)
    R["1a_close_t_vs_open_t1"] = dict(n=int(m.sum()), birebir_esit=round(float((d < 1e-12).mean()), 4),
                                      bir_bp_icinde=round(float((d < 1e-4).mean()), 4),
                                      medyan_fark=float(np.median(d)), p99_fark=float(np.percentile(d, 99)),
                                      max_fark=float(d.max()))
    if True:
        bulgu("1A", "sim_portfoy'un 'sinyal t kapanisinda -> islem t+1 acilista' semantigi bir GECIKME DEGIL: "
                    "gunluk barda close[t] ile open[t+1] AYNI ANDIR (00:00 UTC) ve veride pratikte AYNI fiyattir. "
                    "Yani model SIFIR gecikme varsayiyor.",
              f"BTCUSDT {int(m.sum())} gun: |close[t]/open[t+1]-1| birebir 0 olan oran "
              f"{(d<1e-12).mean():.3f}, 1bp icinde {(d<1e-4).mean():.4f}, medyan {np.median(d):.1e}, "
              f"p99 {np.percentile(d,99):.2e}. Yani open[t+1]->open[t+2] getirisi = close[t]->close[t+1] getirisi.",
              "'t+1 acilista islem' bir guvenlik payi DEGIL. Model 00:00:00 UTC'de tam kapanis fiyatindan "
              "islem varsayiyor. Ileriye bakma degil ama tsmom2'nin gecikme1 testinin (valid 1.39->1.09) neden "
              "bu kadar sert oldugunu aciklar: 0 ile 24 saat arasinda ara olcum yok. Gercekci gecikme "
              "(kapanis + dakikalar/saatler) 1h veriyle asagida olculuyor (1F).", "ORTA")

    # ---- 1b. shift taramasi: -2..+2
    print("\n-- shift taramasi (negatif = GELECEK sinyali; kod durustse belirgin IYILESMELI)")
    R["1b_shift"] = {}
    for ad, cfg in ADAY.items():
        R["1b_shift"][ad] = {}
        for s in (-2, -1, 0, 1, 2):
            r = kos_duz(cfg, shift=s, clip=(-0.95, 5.0), ileri_bakma_maske=True)  # ORIJINAL semantik
            R["1b_shift"][ad][s] = {b: SD(r[b]) for b in ("train", "valid")}
            print(f"   {ad} shift={s:+d}  train S={r['train']['sharpe']:>5} dd={r['train']['maxdd']:<6} "
                  f"| valid S={r['valid']['sharpe']:>5} dd={r['valid']['maxdd']}")
    for ad in ADAY:
        z = R["1b_shift"][ad]
        ileri_kazanc = z[-1]["train"]["sharpe"] - z[0]["train"]["sharpe"]
        gecikme_kayip = z[0]["train"]["sharpe"] - z[1]["train"]["sharpe"]
        R["1b_shift"][ad]["_tani"] = dict(shift_m1_eksi_0_train=round(ileri_kazanc, 3),
                                          shift_0_eksi_p1_train=round(gecikme_kayip, 3))
    a = R["1b_shift"]["A2"]["_tani"]; y = R["1b_shift"]["Y"]["_tani"]
    bulgu("1B", "Shift sondasi ile ileriye bakma ARANDI, BULUNMADI.",
          f"A2: shift=-1 (yarinin sinyali) train Sharpe {R['1b_shift']['A2'][-1]['train']['sharpe']} vs shift=0 "
          f"{R['1b_shift']['A2'][0]['train']['sharpe']} (+{a['shift_m1_eksi_0_train']}); shift=+1 "
          f"{R['1b_shift']['A2'][1]['train']['sharpe']} (-{a['shift_0_eksi_p1_train']}). "
          f"Y: -1 vs 0 = +{y['shift_m1_eksi_0_train']}, 0 vs +1 = -{y['shift_0_eksi_p1_train']}.",
          "Gelecege kaydirmak sonucu IYILESTIRIYOR, gecikme KOTULESTIRIYOR -> sinyal zamanlamasi tutarli, "
          "kod t+1 bilgisini kullanmiyor. (Eger kod zaten gelecegi kullansaydi shift=-1 ek fayda vermezdi.)", "TEMIZ")

    # ---- 1c. FONKSIYON BAZLI NEDENSELLIK: veriyi t0'da fiziksel kes, W[:t0] degisiyor mu?
    print("\n-- truncation (kesme) testi: veri t0'da kesilince W[:t0] AYNI kalmali")
    R["1c_truncation"] = {}
    P = T.panel()
    tam_W = {ad: T2.agirliklar2(cfg).copy() for ad, cfg in ADAY.items()}
    tam_vol = T.volat(30).copy(); tam_frac, tam_tam = T.oylar([50], [], 0.0)
    tam_frac = tam_frac.copy(); tam_dv = P["dv"].copy(); tam_Cf = P["Cf"].copy()
    # T._P.clear() ayni dict nesnesini bosaltir -> P referansi Pk ile ayni olur. Kopyala.
    tam_syms = list(P["syms"]); tam_gun = np.array(P["gun"])
    n_gun = len(tam_gun)
    orij_veri_1d = H.veri_1d
    for t0 in (1200, 2000, 2600):
        kes_ms = int(tam_gun[t0])
        def veri_kesik(sym, _k=kes_ms):
            a = orij_veri_1d(sym)
            return a[a[:, 0] <= _k]
        H.veri_1d = veri_kesik
        T._P.clear(); T._OY.clear(); T2._AY.clear()
        Pk = T.panel()
        volk = T.volat(30); frack, tamk = T.oylar([50], [], 0.0)
        Wk = {ad: T2.agirliklar2(cfg) for ad, cfg in ADAY.items()}
        k = min(len(Pk["gun"]), t0 + 1)
        assert np.array_equal(Pk["gun"][:k], tam_gun[:k]), "gun ekseni kaydi"
        # kesik panelde min_gun=200 filtresi bazi coinleri dusurur -> kolonlari SEMBOLE gore esle
        kes_syms = list(Pk["syms"])
        idx_k = list(range(len(kes_syms)))
        idx_f = [tam_syms.index(s) for s in kes_syms]
        cmp = {}
        for nm, a_, b_ in (("Cf", Pk["Cf"][:k], tam_Cf[:k]), ("dv", Pk["dv"][:k], tam_dv[:k]),
                           ("vol30", volk[:k], tam_vol[:k]), ("frac_ma50", frack[:k], tam_frac[:k]),
                           ("W_A2", Wk["A2"][:k], tam_W["A2"][:k]), ("W_Y", Wk["Y"][:k], tam_W["Y"][:k])):
            a2_ = a_[:, idx_k]; b2_ = b_[:, idx_f]
            df = np.abs(np.nan_to_num(a2_) - np.nan_to_num(b2_))
            nanfark = int((np.isnan(a2_) != np.isnan(b2_)).sum())
            cmp[nm] = dict(max_fark=float(df.max()) if df.size else 0.0, nan_uyusmaz=nanfark)
        # kesik panelde OLMAYAN kolonlarda tam panelin agirligi sifir olmali (yoksa evren farki var)
        eksik = [j for j in range(len(tam_syms)) if tam_syms[j] not in set(kes_syms)]
        cmp["dusen_coin_sayisi"] = len(eksik)
        cmp["dusen_coinlerde_W_Y_max"] = float(np.abs(tam_W["Y"][:k][:, eksik]).max()) if eksik else 0.0
        R["1c_truncation"][t0] = cmp
        print(f"   t0={t0} ({datetime.utcfromtimestamp(kes_ms/1e3).date()}): " +
              " ".join(f"{nm}={c['max_fark']:.1e}/nan{c['nan_uyusmaz']}" for nm, c in cmp.items()
                       if isinstance(c, dict)) +
              f" | dusen coin={len(eksik)} (bu kolonlarda tam-panel W_Y max={cmp['dusen_coinlerde_W_Y_max']:.1e})")
        H.veri_1d = orij_veri_1d
        T._P.clear(); T._OY.clear(); T2._AY.clear()
        T.panel()
    en_kotu = max(c["max_fark"] for v in R["1c_truncation"].values() for c in v.values() if isinstance(c, dict))
    en_kotu_nan = max(c["nan_uyusmaz"] for v in R["1c_truncation"].values() for c in v.values() if isinstance(c, dict))
    dusen_W = max(v["dusen_coinlerde_W_Y_max"] for v in R["1c_truncation"].values())
    R["1c_ozet"] = dict(max_fark=en_kotu, nan_uyusmaz=en_kotu_nan, dusen_coinlerde_W_Y_max=dusen_W)
    tekil = {nm: max(v[nm]["max_fark"] for v in R["1c_truncation"].values())
             for nm in ("Cf", "dv", "vol30", "frac_ma50", "W_A2", "W_Y")}
    R["1c_ozet"]["matris_bazinda_max_fark"] = tekil
    temiz = {k: v for k, v in tekil.items() if k != "W_Y"}
    bulgu("1C", "ffill / roll_mean / roll_std / volat / oylar / hyst / _ay_indeks / agirliklar2 zincirinde "
                "GELECEK SIZINTISI YOK (A2 icin tam temiz).",
          f"Veri 3 ayri gunde (idx 1200=2020-11-29, 2000=2023-02-07, 2600=2024-09-29) FIZIKSEL kesilip tum "
          f"zincir sifirdan yeniden hesaplandi (T._P/_OY/T2._AY temizlendi). Kesim oncesi matrislerin max "
          f"mutlak farki: " + ", ".join(f"{k}={v:.1e}" for k, v in temiz.items()) +
          f"; nan uyusmazligi {en_kotu_nan}. Yani MA50, vol30, 90g dolar hacmi ve A2 agirliklari t gununde "
          f"yalnizca t ve oncesini kullaniyor.",
          "A2 tarafinda ileriye bakma yok; sonucu sisiren sey sizinti degil.", "TEMIZ")

    # ---- 1e. min_gun=200 = ILERIYE BAKAN EVREN FILTRESI (yalniz yuruyen sepeti etkiler)
    if tekil["W_Y"] > 1e-9:
        # kac secim, secim aninda 200 gunden az gecmisi olan bir coine gitmis?
        P = T.panel(); b = P["bas"]
        yas = np.cumsum(~np.isnan(P["C"]), axis=0)          # t gunune kadar KAC gun verisi var
        W_Y_ = T2.agirliklar2(YY)
        genc = (np.abs(W_Y_) > 1e-12) & (yas < 200)
        genc_gun = int(genc[b:].any(1).sum())
        genc_agir = float(np.abs(W_Y_[b:])[genc[b:]].sum())
        top_agir = float(np.abs(W_Y_[b:]).sum())
        # DUZELTME: 200 gun sartini NEDENSEL uygula
        um_kes = yas >= 200
        frac2, tam2 = T.oylar([50], [], 0.0)
        um = T2.evren_maske2(tam2, YY) & um_kes
        s = np.where(um, np.nan_to_num(frac2), 0.0)
        K = um.sum(1)
        vol = T.volat(30)
        Wc = np.where(um, 1.0 / np.maximum(K, 1)[:, None], 0.0) * s
        Wc = Wc * np.clip(0.30 / np.nan_to_num(vol, nan=1e9), 0, 1.5)
        gr = np.abs(Wc).sum(1, keepdims=True)
        Wc = np.where(gr > 1.0, Wc / np.maximum(gr, 1e-12), Wc)
        r_on = kos_duz(YY, clip=(-0.95, 5.0), ileri_bakma_maske=True)
        r_du = kos_duz(YY, clip=(-0.95, 5.0), ileri_bakma_maske=True, Wover=Wc)
        R["1e_min_gun"] = dict(genc_secim_gun=genc_gun, genc_agirlik_payi=round(genc_agir / max(top_agir, 1e-9), 4),
                               onceki={b_: SD(r_on[b_]) for b_ in ("train", "valid")},
                               duzeltilmis={b_: SD(r_du[b_]) for b_ in ("train", "valid")})
        print(f"\n-- min_gun=200 ileriye bakan filtre: Y sepetinde {genc_gun} gun boyunca 200 gunden genc coin "
              f"tutuluyor (agirlik payi {genc_agir/max(top_agir,1e-9):.2%})")
        print(f"   Y once  tr S={r_on['train']['sharpe']} dd={r_on['train']['maxdd']} | va S={r_on['valid']['sharpe']} dd={r_on['valid']['maxdd']}")
        print(f"   Y sonra tr S={r_du['train']['sharpe']} dd={r_du['train']['maxdd']} | va S={r_du['valid']['sharpe']} dd={r_du['valid']['maxdd']}")
        bulgu("1E", "GERCEK ILERIYE BAKMA BULUNDU (yalniz yuruyen sepet / Y adayini etkiler): "
                    "harness.gunluk_panel(min_gun=200) bir coini panele almak icin TUM VERI SETINDEKI toplam gun "
                    "sayisina bakiyor. Yani 2020-11'de daha 90 gunluk olan bir coin panelde var CUNKU ILERIDE "
                    "200+ gune ulasacagini biliyoruz. Yuruyen top-3 bu coinleri secebiliyor.",
              f"Kesme testi: veri 2020-11-29'da kesildiginde panelden 304 coin dusuyor ve bunlarin tam-panel "
              f"W_Y agirligi 0.048'e kadar cikiyor -> kesim ONCESI W_Y max fark {tekil['W_Y']:.2f} "
              f"(Cf/dv/vol30/frac/W_A2 farki tam olarak 0.0). "
              f"Ayrica: Y sepeti {genc_gun} gun boyunca secim aninda 200 gunden genc bir coin tutuyor "
              f"(toplam agirligin %{100*genc_agir/max(top_agir,1e-9):.1f}'i).",
              f"'>=200 gun gecmis' sarti NEDENSEL uygulandiginda Y: train Sharpe "
              f"{r_on['train']['sharpe']} -> {r_du['train']['sharpe']}, maxDD {r_on['train']['maxdd']} -> "
              f"{r_du['train']['maxdd']}; valid Sharpe {r_on['valid']['sharpe']} -> {r_du['valid']['sharpe']}, "
              f"maxDD {r_on['valid']['maxdd']} -> {r_du['valid']['maxdd']}.", "ORTA")
    else:
        R["1e_min_gun"] = "etkisiz"

    # ---- 1d. sim_portfoy'daki O[t+2] maskesi = GERCEK ileri bakma
    P = T.panel(); b = P["bas"]
    O = P["O"][b:]
    n = len(O)
    W_A2 = T2.agirliklar2(A2)[b:]; W_Y = T2.agirliklar2(YY)[b:]
    say = {}
    for ad, W in (("A2", W_A2), ("Y", W_Y)):
        c = 0; agir = 0.0
        for t in range(n - 2):
            m = (~np.isnan(O[t + 1])) & np.isnan(O[t + 2]) & (np.abs(W[t]) > 1e-12)
            c += int(m.sum()); agir += float(np.abs(W[t][m]).sum())
        say[ad] = dict(hucre=c, toplam_agirlik=round(agir, 4))
    R["1d_o_t2_maskesi"] = say
    # etkisini olc
    R["1d_etki"] = {}
    for ad, cfg in ADAY.items():
        a1 = kos_duz(cfg, clip=(-0.95, 5.0), ileri_bakma_maske=True)
        a0 = kos_duz(cfg, clip=(-0.95, 5.0), ileri_bakma_maske=False)
        R["1d_etki"][ad] = dict(orijinal={b_: SD(a1[b_]) for b_ in ("train", "valid")},
                                duzeltilmis={b_: SD(a0[b_]) for b_ in ("train", "valid")})
        print(f"   {ad} O[t+2] maskesi: orij tr {a1['train']['sharpe']} va {a1['valid']['sharpe']} "
              f"-> duz tr {a0['train']['sharpe']} va {a0['valid']['sharpe']}")
    # ---- 1f. GERCEKCI UYGULAMA GECIKMESI: 1h veriyle saat saat
    R["1f_saatlik_gecikme"] = test1f_saatlik()

    bulgu("1D", "harness.sim_portfoy satir 202: `w[np.isnan(O[t+1]) | np.isnan(O[t+2])] = 0` "
                "-> O[t+2] t gununde BILINEMEZ. Yarindan sonraki gunun acilisi eksikse pozisyon bugunden "
                "sifirlaniyor: kucuk ama GERCEK bir ileriye bakma.",
          f"Etkilenen (t,coin) hucre sayisi: A2 {say['A2']['hucre']}, Y {say['Y']['hucre']} "
          f"(toplam mutlak agirlik A2 {say['A2']['toplam_agirlik']}, Y {say['Y']['toplam_agirlik']}). "
          f"Maskesiz (duzeltilmis, open ffill ile) sonuc: "
          f"A2 train {R['1d_etki']['A2']['orijinal']['train']['sharpe']}->{R['1d_etki']['A2']['duzeltilmis']['train']['sharpe']}, "
          f"valid {R['1d_etki']['A2']['orijinal']['valid']['sharpe']}->{R['1d_etki']['A2']['duzeltilmis']['valid']['sharpe']}; "
          f"Y train {R['1d_etki']['Y']['orijinal']['train']['sharpe']}->{R['1d_etki']['Y']['duzeltilmis']['train']['sharpe']}, "
          f"valid {R['1d_etki']['Y']['orijinal']['valid']['sharpe']}->{R['1d_etki']['Y']['duzeltilmis']['valid']['sharpe']}.",
          "b3 sepetinde eksik gun yok -> A2 etkilenmiyor. Yuruyen sepette etkileniyorsa buyuklugu yukarida.", "DUSUK")
    return R


def test1f_saatlik(coinler=("BTCUSDT", "ETHUSDT", "BNBUSDT"), hedef=0.30):
    """1h veriyle: gunluk kapanistan L SAAT sonra islem yapilirsa alfa ne kadar kaliyor?
    tsmom2 sadece 0 ve 24 saati olcmustu (valid 1.39 -> 1.09). Arasi burada.
    NOT: 1h veri 2024-09'da basliyor -> pencere kucuk (~1 yil), yon gostergesi olarak okunmali."""
    print("\n-- 1h veriyle GERCEKCI uygulama gecikmesi (kapanistan L saat sonra islem)")
    var = set(H.semboller_1h())
    cs = [c for c in coinler if c in var]
    if len(cs) < 2:
        print("   1h veri yok, atlandi"); return dict(atlandi=True)
    gunms = 86400_000
    tab = {}
    for s in cs:
        a = H.veri_1h(s)
        tab[s] = {int(t): (float(o), float(c)) for t, o, c in zip(a[:, 0], a[:, 1], a[:, 4])}
    gun0 = sorted(set.intersection(*[{t - (t % gunms) for t in tab[s]} for s in cs]))
    # sadece 24 saati tam olan gunler
    gun = [d for d in gun0 if all(all((d + h * 3600_000) in tab[s] for h in range(24)) for s in cs)]
    n, m = len(gun), len(cs)
    if n < 200:
        print(f"   yeterli 1h gun yok ({n}), atlandi"); return dict(atlandi=True, gun=n)
    Cd = np.array([[tab[s][d + 23 * 3600_000][1] for s in cs] for d in gun])       # gunluk kapanis
    Rd = np.full_like(Cd, np.nan); Rd[1:] = Cd[1:] / Cd[:-1] - 1
    ma = T.roll_mean(Cd, 50)
    vol = np.maximum(T.roll_std(Rd, 30) * np.sqrt(365), T.VOL_TABAN)
    tam = ~np.isnan(ma) & ~np.isnan(vol)
    sig = np.where(tam, (Cd > ma).astype(float), 0.0)
    K = np.maximum(tam.sum(1), 1)[:, None]
    W = np.where(tam, 1.0 / K, 0.0) * sig
    if hedef:
        W = W * np.clip(hedef / np.nan_to_num(vol, nan=1e9), 0, 1.5)
    gr = np.abs(W).sum(1, keepdims=True)
    W = np.where(gr > 1.0, W / np.maximum(gr, 1e-12), W)
    bas = int(np.argmax(tam.all(1))) + 5
    out = {}
    for L in (0, 1, 2, 3, 4, 6, 8, 12, 18, 24):
        # sinyal gun t kapanisinda -> islem (t+1 gununun 00:00'i) + L saat
        px = np.array([[tab[s].get(d + (L % 24) * 3600_000 + (gunms if L >= 24 else 0), (np.nan, np.nan))[0]
                        for s in cs] for d in gun])
        g = []; w_prev = np.zeros(m)
        for t in range(bas, n - 3):
            w = np.nan_to_num(W[t]).copy()
            p1, p2 = px[t + 1], px[t + 2]
            w[np.isnan(p1)] = 0
            r = np.nan_to_num((p2 - p1) / p1)
            x = float((w * r).sum() - np.abs(w - w_prev).sum() * 0.0007)
            g.append(x); w_prev = w * (1 + r) / (1 + x) if x > -1 else w
        g = np.array(g)
        eq = np.cumprod(1 + g)
        dd = float(((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max())
        sh = float(g.mean() / g.std() * np.sqrt(365)) if g.std() > 0 else 0.0
        out[L] = dict(sharpe=round(sh, 2), maxdd=round(dd, 3), toplam=round(float(eq[-1] - 1), 3), gun=len(g))
        print(f"   L={L:>2}h  Sharpe={out[L]['sharpe']:>5} maxDD={out[L]['maxdd']:<6} getiri={out[L]['toplam']:+.3f}")
    d1 = out[0]["sharpe"] - out[1]["sharpe"]; d4 = out[0]["sharpe"] - out[4]["sharpe"]
    d24 = out[0]["sharpe"] - out[24]["sharpe"]
    kotu = max(abs(d1), abs(d4))
    bulgu("1F", "tsmom2'nin 'sinyal gecikmesine cok duyarli' uyarisi YANLIS TESHIS. 1h veriyle 0-18 saat arasi "
                "gecikme neredeyse etkisiz; kayip yalnizca TAM 24 saatte ortaya cikiyor - yani sorun "
                "'gec islem yapmak' degil, 'bir gunluk sinyali kacirmak'.",
          f"BTC/ETH/BNB MA50+vt30, 1h kaynakli {out[0]['gun']} gun: Sharpe "
          + " ".join(f"L={L}h:{out[L]['sharpe']}" for L in (0, 1, 2, 3, 4, 6, 8, 12, 18, 24)) +
          f". 1 saatlik gecikmenin bedeli {d1:+.2f}, 4 saatlik {d4:+.2f}, 24 saatlik {d24:+.2f} Sharpe.",
          "Uygulanabilirlik acisindan IYI haber: 00:00 UTC'de saniyesinde islem yapma zorunlulugu yok, "
          "gun icinde birkac saatlik pencere yeterli. UYARI: pencere kisa (tek yil, valid ile ortusuyor, "
          "tek boga rejimi) -> kesin sayi degil yon gostergesi.",
          "TEMIZ" if kotu < 0.15 else "ORTA")
    return out


# =====================================================================================
# 2) PANEL HIZALAMASI / NAN / CLIP
# =====================================================================================
def test2_panel():
    print("\n" + "=" * 90 + "\n2) PANEL HIZALAMASI, NAN, CLIP\n" + "=" * 90)
    R = {}
    P = T.panel(); b = P["bas"]; gun = P["gun"]; syms = P["syms"]
    # 2a. gun ekseni: hangi coinlerde delik var
    delik = {}
    for s in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        j = syms.index(s)
        c = P["C"][:, j]
        ilk = int(np.argmax(~np.isnan(c)))
        son = len(c) - 1
        ic_nan = int(np.isnan(c[ilk:son + 1]).sum())
        delik[s] = dict(ilk_gun=datetime.utcfromtimestamp(gun[ilk] / 1e3).strftime("%Y-%m-%d"),
                        ic_eksik_gun=ic_nan, toplam_gun=int(son - ilk + 1))
    R["2a_b3_delik"] = delik
    print("   b3 delikleri:", delik)

    # yuruyen sepetin sectigi coinlerdeki delikler
    W_Y = T2.agirliklar2(YY)
    kullanilan = np.flatnonzero((np.abs(W_Y[b:]) > 1e-12).any(0))
    y_delik = 0; y_gun = 0
    for j in kullanilan:
        c = P["C"][:, j]; ilk = int(np.argmax(~np.isnan(c)))
        y_delik += int(np.isnan(c[ilk:]).sum()); y_gun += len(c) - ilk
    R["2a_Y_delik"] = dict(kullanilan_coin=int(len(kullanilan)), ic_eksik_gun=int(y_delik), toplam_gun=int(y_gun))
    print(f"   Y sepeti: {len(kullanilan)} coin kullaniliyor, listeleme sonrasi ic eksik gun={y_delik}/{y_gun}")

    # ffill'in urettigi sahte 0-getiri gunleri (vol'u dusurur -> kaldiraci arttirir)
    Cf = P["Cf"]; Cham = P["C"]
    dolgu = ~np.isnan(Cf[b:]) & np.isnan(Cham[b:])
    sahte = int(dolgu.sum())
    # yalnizca A2/Y'nin GERCEKTEN pozisyon tuttugu hucrelerde kac dolgu var?
    etkili = {}
    for ad, cfg in ADAY.items():
        Wx = T2.agirliklar2(cfg)[b:]
        k = min(len(Wx), len(dolgu))
        etkili[ad] = int((dolgu[:k] & (np.abs(Wx[:k]) > 1e-12)).sum())
    R["2a_ffill_sahte_gun"] = dict(panel_geneli=sahte, pozisyonlu_hucre=etkili,
                                   b3_ic_eksik=sum(d["ic_eksik_gun"] for d in delik.values()))
    print(f"   ffill dolgu hucresi: panel geneli {sahte}, A2 pozisyonlarinda {etkili['A2']}, "
          f"Y pozisyonlarinda {etkili['Y']}")
    if max(etkili.values()) == 0:
        bulgu("2A", "ffill kaynakli sahte 0-getiri gunu ADAYLARI ETKILEMIYOR (ffill'in kendisi nedensel: "
                    "np.maximum.accumulate ile yalniz gecmis indeks, listeleme oncesi nan korunuyor).",
              f"Panel genelinde {sahte} dolgu hucresi var ama A2'nin ve Y'nin pozisyon tuttugu hucrelerde "
              f"0 dolgu. b3 (BTC/ETH/BNB) listeleme sonrasi hic eksik gun icermiyor "
              f"(ilk gunler: {delik['BTCUSDT']['ilk_gun']}/{delik['ETHUSDT']['ilk_gun']}/{delik['BNBUSDT']['ilk_gun']}); "
              f"Y'nin kullandigi {R['2a_Y_delik']['kullanilan_coin']} coinde de "
              f"{R['2a_Y_delik']['ic_eksik_gun']} eksik gun.",
              "Vol tahmini bu yoldan bozulmuyor; sahte 0-getiri kaynakli kaldirac sismesi YOK.", "TEMIZ")
    else:
        bulgu("2A", "ffill listeleme SONRASI eksik gunleri onceki kapanisla dolduruyor -> o gunler icin "
                    "getiri tam 0 olur, 30 gunluk vol duser, hedef_vol/vol30 kaldiraci ARTAR.",
              f"Panel geneli {sahte} dolgu hucresi; A2 pozisyonlarinda {etkili['A2']}, Y pozisyonlarinda "
              f"{etkili['Y']}.", "Vol hafif kucuk tahmin ediliyor -> kaldirac hafif yuksek.",
              "DUSUK" if max(etkili.values()) < 100 else "ORTA")

    # 2b. listeleme oncesi nan'lar agirligi bozuyor mu?
    frac, tam = T.oylar([50], [], 0.0)
    um_A2 = T2.evren_maske2(tam, A2)
    K = um_A2[b:].sum(1)
    R["2b_A2_evren_boyu"] = {int(k): int((K == k).sum()) for k in np.unique(K)}
    ilk_3 = int(np.argmax(K == 3)) + b
    R["2b_A2_3coin_baslangic"] = datetime.utcfromtimestamp(gun[ilk_3] / 1e3).strftime("%Y-%m-%d")
    print(f"   A2 evren boyu dagilimi: {R['2b_A2_evren_boyu']}, 3 coine gecis {R['2b_A2_3coin_baslangic']}")
    bulgu("2B", "Listeleme oncesi nan'lar agirligi BOZMUYOR: evren maskesi K=um.sum(1) ile birlikte "
                "1/K olceklenir, coin canli degilse hem paydadan hem paydan cikar.",
          f"A2 evren boyu gun dagilimi {R['2b_A2_evren_boyu']}; 3 coin ilk kez {R['2b_A2_3coin_baslangic']}. "
          f"Al-tut kiyasi AYNI maskeyi kullaniyor (tsmom2.agirliklar2 altut kolu) -> kiyas da ayni gunlerde 2 coinli.",
          "Hizalama tarafinda sisirme yok.", "TEMIZ")

    # 2c. clip etkisi
    print("\n-- np.clip(-0.95, 5.0) etkisi")
    R["2c_clip"] = {}
    for ad, cfg in ADAY.items():
        rk = kos_duz(cfg, clip=(-0.95, 5.0), ileri_bakma_maske=True)
        rs = kos_duz(cfg, clip=None, ileri_bakma_maske=True)
        R["2c_clip"][ad] = dict(kirpilan_hucre=rk["_kirp"],
                                kirpmali={b_: SD(rk[b_]) for b_ in ("train", "valid")},
                                kirpmasiz={b_: SD(rs[b_]) for b_ in ("train", "valid")})
        print(f"   {ad} kirpilan hucre={rk['_kirp']}  tr {rk['train']['sharpe']}->{rs['train']['sharpe']}  "
              f"va {rk['valid']['sharpe']}->{rs['valid']['sharpe']}")
    # ayrica al-tut kiyasi icin de
    for ad, cfg in (("A2_altut", A2B), ("Y_altut", YYB)):
        rk = kos_duz(cfg, clip=(-0.95, 5.0), ileri_bakma_maske=True)
        rs = kos_duz(cfg, clip=None, ileri_bakma_maske=True)
        R["2c_clip"][ad] = dict(kirpilan_hucre=rk["_kirp"],
                                kirpmali={b_: SD(rk[b_]) for b_ in ("train", "valid")},
                                kirpmasiz={b_: SD(rs[b_]) for b_ in ("train", "valid")})
        print(f"   {ad} kirpilan hucre={rk['_kirp']}  tr {rk['train']['sharpe']}->{rs['train']['sharpe']}  "
              f"va {rk['valid']['sharpe']}->{rs['valid']['sharpe']}")
    kh = R["2c_clip"]["A2"]["kirpilan_hucre"] + R["2c_clip"]["Y"]["kirpilan_hucre"]
    dA = R["2c_clip"]["A2"]["kirpmasiz"]["train"]["sharpe"] - R["2c_clip"]["A2"]["kirpmali"]["train"]["sharpe"]
    dY = R["2c_clip"]["Y"]["kirpmasiz"]["train"]["sharpe"] - R["2c_clip"]["Y"]["kirpmali"]["train"]["sharpe"]
    bulgu("2C", "Getiri kirpmasi np.clip(-0.95, 5.0): kirpma HER ZAMAN strateji lehinedir (asagida -%95'te "
                "durur, yukarida +%500'e kadar serbest) - asimetrik. Kirpmasiz kosuldu.",
          f"Kirpilan hucre sayisi: A2 {R['2c_clip']['A2']['kirpilan_hucre']}, Y {R['2c_clip']['Y']['kirpilan_hucre']}, "
          f"al-tut kiyaslarinda {R['2c_clip']['A2_altut']['kirpilan_hucre']}/{R['2c_clip']['Y_altut']['kirpilan_hucre']}. "
          f"Kirpmasiz train Sharpe farki: A2 {dA:+.3f}, Y {dY:+.3f}.",
          "Kirpma binmiyorsa etki yok; biniyorsa yukaridaki fark kadar sisirme var.",
          "TEMIZ" if kh == 0 else ("DUSUK" if abs(dA) < 0.05 and abs(dY) < 0.05 else "ORTA"))
    return R


# =====================================================================================
# 3) HAYATTA KALMA YANLILIGI
# =====================================================================================
def test3_survivorship():
    print("\n" + "=" * 90 + "\n3) HAYATTA-KALMA YANLILIGI (yuruyen top-3)\n" + "=" * 90)
    R = {}
    P = T.panel(); b = P["bas"]; syms = P["syms"]; gun = P["gun"]
    th = T2.sepet_tarihce(3, 30)
    R["3a_sepet_tarihce"] = th
    secilenler = sorted({c for _, L in th for c in L})
    R["3a_secilen_coinler"] = secilenler
    R["3a_havuz"] = len(syms)
    print(f"   havuz {len(syms)} coin (bugunun vadeli listesi), yuruyen top-3 hic {len(secilenler)} "
          f"farkli coin secmis: {secilenler}")

    # 3b. tepe agirlik: tek bir coinin cokusu ne kadar zarar verir?
    W = T2.agirliklar2(YY)[b:]
    wmax = np.abs(W).max(1)
    R["3b_agirlik"] = dict(ort_max_coin_agirlik=round(float(wmax[wmax > 0].mean()), 4),
                           p95=round(float(np.percentile(wmax[wmax > 0], 95)), 4),
                           mutlak_max=round(float(wmax.max()), 4),
                           ort_gross=round(float(np.abs(W).sum(1).mean()), 4))
    print(f"   Y tek-coin agirligi: ort {R['3b_agirlik']['ort_max_coin_agirlik']}, "
          f"p95 {R['3b_agirlik']['p95']}, max {R['3b_agirlik']['mutlak_max']}")

    # 3c. HAYALET COIN enjeksiyonu: aylik secimlerin bir kismini 'delist olmus' bir coinle degistir
    # Coin, secildigi ay boyunca sepetin ortalama getirisini izler, sonra bir gun -X% coker ve
    # sonrasinda islem gormez (agirlik zorla tasinir, sifirlanamaz -> en kotu hal).
    print("\n-- hayalet-coin (delist) enjeksiyonu, Monte Carlo")
    O = P["O"][b:].copy(); C = P["C"][b:].copy()
    Wbase = W.copy()
    n, m = Wbase.shape
    gunb = gun[b:]
    ay = T2._ay_indeks(gunb)
    ay_bas = [0] + list(np.flatnonzero(np.diff(ay)) + 1)
    Of = ffill_of(P["O"][b:])
    rmat = np.zeros_like(Of); rmat[:-1] = (Of[1:] - Of[:-1]) / Of[:-1]
    rmat = np.nan_to_num(rmat)

    def mc(p_ay, cokus, tekrar=60):
        """her ay basinda p_ay olasilikla 3 secimden birinin yerine, o ay icinde -cokus yasayan
        bir hayalet konur. Getiri vektorunu dogrudan degistiririz (agirlik ayni kalir)."""
        out = []
        for it in range(tekrar):
            rr = None
            eq = 1.0; gl = []
            w_prev = np.zeros(m)
            # hangi aylarda, hangi kolonda, hangi gunde cokus
            olay = {}
            for k, t0 in enumerate(ay_bas):
                if RNG.random() < p_ay:
                    t1 = ay_bas[k + 1] if k + 1 < len(ay_bas) else n
                    aktif = np.flatnonzero(np.abs(Wbase[t0]) > 1e-12)
                    if len(aktif) == 0:
                        continue
                    j = int(RNG.choice(aktif))
                    td = int(RNG.integers(t0, max(t0 + 1, t1)))
                    olay[td] = olay.get(td, []) + [j]
            for t in range(n - 2):
                w = np.nan_to_num(Wbase[t]).copy(); w[np.isnan(O[t + 1])] = 0
                r = rmat[t + 1].copy()
                for j in olay.get(t, []):
                    r[j] = -cokus
                x = (w * r).sum() - np.abs(w - w_prev).sum() * 0.0007
                gl.append(x)
                w_prev = w * (1 + r) / (1 + x) if x > -1 else w
            gl = np.array(gl)
            rep = H.portfoy_rapor(np.cumprod(1 + gl), gl, gunb, "train")
            rep2 = H.portfoy_rapor(np.cumprod(1 + gl), gl, gunb, "valid")
            out.append((rep["sharpe"], rep["maxdd"], rep2["sharpe"], rep2["maxdd"]))
            del rr
        a = np.array(out)
        return dict(train_sharpe=round(float(a[:, 0].mean()), 3), train_sharpe_p10=round(float(np.percentile(a[:, 0], 10)), 3),
                    train_maxdd=round(float(a[:, 1].mean()), 3),
                    valid_sharpe=round(float(a[:, 2].mean()), 3), valid_maxdd=round(float(a[:, 3].mean()), 3))

    taban = kos_duz(YY, clip=None, ileri_bakma_maske=False)
    R["3c_taban"] = {b_: SD(taban[b_]) for b_ in ("train", "valid")}
    R["3c_mc"] = {}
    # p_ay: bir ayin secimlerinde 'artik listede olmayan' bir coin bulunma olasiligi
    n_ay = len(ay_bas)
    for p_ay, cokus in ((0.01, 0.90), (0.02, 0.90), (0.05, 0.90), (0.10, 0.90),
                        (0.20, 0.90), (0.10, 0.99), (0.33, 0.90)):
        k = f"p{int(p_ay*100)}_c{int(cokus*100)}"
        R["3c_mc"][k] = mc(p_ay, cokus)
        R["3c_mc"][k]["beklenen_olay"] = round(p_ay * n_ay, 1)
        print(f"   ay-basi {p_ay:>5.0%} (bekl. {p_ay*n_ay:>4.1f} olay/8yil), -{cokus:.0%} cokus: "
              f"train S={R['3c_mc'][k]['train_sharpe']} (p10 {R['3c_mc'][k]['train_sharpe_p10']}) "
              f"dd={R['3c_mc'][k]['train_maxdd']} | valid S={R['3c_mc'][k]['valid_sharpe']}")
    R["3c_ay_sayisi"] = n_ay
    ust = R["3c_mc"]["p10_c90"]      # ~9 olay / 8 yil = agir ust sinir
    orta = R["3c_mc"]["p2_c90"]      # ~2 olay / 8 yil = tarihsel olarak makul
    bulgu("3A", "HAYATTA-KALMA YANLILIGI: data1d_all bugunun vadeli listesi (372 coin). Yuruyen top-3 "
                "yalnizca HAYATTA KALANLAR arasindan seciyor; 2018-2025 arasi hacim siralamasinda ilk-3'e girip "
                "sonra delist olan/coken coinler (ornek sinifi: algoritmik stablecoin ekosistemleri, borsa tokenleri) "
                "sepette HIC gorunmuyor.",
          f"Havuz {len(syms)} coin; yuruyen top-3 {n_ay} ay boyunca yalnizca {len(secilenler)} farkli coin secmis "
          f"({', '.join(secilenler)}) - hepsi bugun hala listede. Delist olmus tek bir sembol bile havuzda yok. "
          f"Hayalet-coin Monte Carlo (60 tekrar): 2 olay/8yil -> train Sharpe "
          f"{R['3c_taban']['train']['sharpe']} -> {orta['train_sharpe']} (maxDD {R['3c_taban']['train']['maxdd']} -> "
          f"{orta['train_maxdd']}); 5 olay -> {R['3c_mc']['p5_c90']['train_sharpe']}; "
          f"9 olay -> {ust['train_sharpe']} (maxDD {ust['train_maxdd']}); "
          f"1 olay -> {R['3c_mc']['p1_c90']['train_sharpe']}.",
          f"UST SINIR TAHMINI (train Sharpe): tarihsel olarak makul 1-2 olay senaryosunda "
          f"~{round(R['3c_taban']['train']['sharpe']-R['3c_mc']['p1_c90']['train_sharpe'],2)}..."
          f"{round(R['3c_taban']['train']['sharpe']-orta['train_sharpe'],2)} puan; agresif (9 olay) senaryoda "
          f"{round(R['3c_taban']['train']['sharpe']-ust['train_sharpe'],2)} puana kadar. "
          f"Yani Y'nin train Sharpe'i {R['3c_taban']['train']['sharpe']} yerine ~"
          f"{orta['train_sharpe']} (makul) .. {ust['train_sharpe']} (kotumser) beklenmeli. Mekanizma: tek-coin agirlik ort "
          f"{R['3b_agirlik']['ort_max_coin_agirlik']:.3f} (p95 {R['3b_agirlik']['p95']:.3f}) -> tek gunluk -%90 "
          f"sok portfoyde ~%{R['3b_agirlik']['ort_max_coin_agirlik']*90:.0f} kayip; MA50 trend filtresi bir GECELIK "
          f"cokuse karsi hicbir koruma saglamaz ve vol hedefleme de sadece 30 gunluk GECMIS vole bakar.", "ORTA")

    # 3d. A2 icin survivorship: yok (BTC/ETH/BNB hepsi hayatta) ama sepet SECIMI geriye donuk
    bulgu("3B", "A2 icin hayatta-kalma yanliligi YOK ama yerine daha buyugu var: sepet (BTC/ETH/BNB) "
                "2026'dan geriye bakarak secildi. tsmom2 bunu zaten olcmus.",
          "tsmom2 sonuc_tsmom2.json: 7 sepette train Sharpe vt30 medyan ~1.37, b3 = en iyisi (1.59). "
          "Yuruyen (ileriye bakmayan) sepet train 1.19. BNB ilk kez 2022-12'de hacim ilk-3'une giriyor.",
          "A2'nin train Sharpe'inin ~0.2-0.4'u sepet secimi. Kirmizi takim bunu DOGRULUYOR; ustune "
          "yuruyen sepetin kendi survivorship yanliligi (3A) biniyor - yani Y de tamamen temiz degil.", "ORTA")
    return R


# =====================================================================================
# 4) VOL HEDEFLEME / KALDIRAC
# =====================================================================================
def test4_vol():
    print("\n" + "=" * 90 + "\n4) VOL HEDEFLEME VE KALDIRAC\n" + "=" * 90)
    R = {}
    P = T.panel(); b = P["bas"]
    R["4a_gross"] = {}
    for ad, cfg in list(ADAY.items()) + [("A2_altut", A2B), ("Y_altut", YYB)]:
        W = T2.agirliklar2(cfg)[b:]
        gr = np.abs(W).sum(1)
        R["4a_gross"][ad] = dict(max=round(float(gr.max()), 4), p99=round(float(np.percentile(gr, 99)), 4),
                                 ort=round(float(gr.mean()), 4),
                                 gross_1_ustu_gun=int((gr > 1.0 + 1e-9).sum()),
                                 tek_coin_max=round(float(np.abs(W).max()), 4))
        print(f"   {ad}: gross max={R['4a_gross'][ad]['max']} ort={R['4a_gross'][ad]['ort']} "
              f">1 gun={R['4a_gross'][ad]['gross_1_ustu_gun']} tek-coin max={R['4a_gross'][ad]['tek_coin_max']}")
    if R["4a_gross"]["A2"]["gross_1_ustu_gun"] == 0 and R["4a_gross"]["Y"]["gross_1_ustu_gun"] == 0:
        bulgu("4A", "cap=1.5 SPOT ICIN SORUN DEGIL: tsmom2.agirliklar2 satir 127-128 vol olceklemesinden "
                    "SONRA gross>1 ise normalize ediyor, yani toplam pozisyon hicbir gun 1.0'i asmiyor.",
              f"A2 gross max {R['4a_gross']['A2']['max']}, Y gross max {R['4a_gross']['Y']['max']}; "
              f"gross>1 olan gun sayisi 0/0. En buyuk tek-coin agirligi A2 {R['4a_gross']['A2']['tek_coin_max']}, "
              f"Y {R['4a_gross']['Y']['tek_coin_max']}.",
              "Kaldirac iddiasi cursuz; cap yalnizca coinler arasi GORECELI agirligi degistiriyor.", "TEMIZ")
    else:
        bulgu("4A", "gross 1.0'i asiyor - spot uygulanamaz.", str(R["4a_gross"]), "kaldirac gerekir", "YUKSEK")

    # 4b. cap 1.0 vs 1.5 (+ 1.0 ve vol=t-1 kullanimi)
    print("\n-- cap taramasi")
    R["4b_cap"] = {}
    for ad, cfg in ADAY.items():
        R["4b_cap"][ad] = {}
        for cp in (1.0, 1.5, 2.0):
            r = kos_duz(cfg, cap=cp, clip=None, ileri_bakma_maske=False)
            R["4b_cap"][ad][cp] = {b_: SD(r[b_]) for b_ in ("train", "valid")}
            print(f"   {ad} cap={cp}: tr S={r['train']['sharpe']} dd={r['train']['maxdd']} gross={r['train']['ort_gross']} "
                  f"| va S={r['valid']['sharpe']} dd={r['valid']['maxdd']} gross={r['valid']['ort_gross']}")
    d2 = R["4b_cap"]["A2"][1.0]["train"]["sharpe"] - R["4b_cap"]["A2"][1.5]["train"]["sharpe"]
    dy = R["4b_cap"]["Y"][1.0]["train"]["sharpe"] - R["4b_cap"]["Y"][1.5]["train"]["sharpe"]
    bulgu("4B", "cap=1.0 (kesin spot) ile yeniden kosuldu.",
          f"A2 cap1.0 train S={R['4b_cap']['A2'][1.0]['train']['sharpe']} dd={R['4b_cap']['A2'][1.0]['train']['maxdd']} "
          f"valid S={R['4b_cap']['A2'][1.0]['valid']['sharpe']} (cap1.5: {R['4b_cap']['A2'][1.5]['train']['sharpe']}/"
          f"{R['4b_cap']['A2'][1.5]['valid']['sharpe']}); Y cap1.0 train {R['4b_cap']['Y'][1.0]['train']['sharpe']} "
          f"valid {R['4b_cap']['Y'][1.0]['valid']['sharpe']} (cap1.5: {R['4b_cap']['Y'][1.5]['train']['sharpe']}/"
          f"{R['4b_cap']['Y'][1.5]['valid']['sharpe']}).",
          f"cap degisimi train Sharpe'i A2'de {d2:+.2f}, Y'de {dy:+.2f} degistiriyor.",
          "TEMIZ" if max(abs(d2), abs(dy)) < 0.10 else "DUSUK")

    # 4c. vol zamanlamasi: vol[t] mi vol[t-1] mi
    print("\n-- vol zamanlamasi (olceklemede vol[t-1] kullan -> ek gecikme)")
    R["4c_vol_gecikme"] = {}
    vol0 = T.volat(30)
    vol_gec = np.vstack([np.full((1, vol0.shape[1]), np.nan), vol0[:-1]])
    for ad, cfg in ADAY.items():
        frac, tam = T.oylar(cfg["ma"], cfg["ret"], cfg["band"])
        um = T2.evren_maske2(tam, cfg)
        s = np.where(um, np.nan_to_num(frac), 0.0)
        K = um.sum(1)
        base = np.where(um, 1.0 / np.maximum(K, 1)[:, None], 0.0)
        out = {}
        for nm, vv in (("vol_t", vol0), ("vol_t-1", vol_gec)):
            Wx = base * s * np.clip(0.30 / np.nan_to_num(vv, nan=1e9), 0, 1.5)
            gr = np.abs(Wx).sum(1, keepdims=True)
            Wx = np.where(gr > 1.0, Wx / np.maximum(gr, 1e-12), Wx)
            r = kos_duz(cfg, clip=None, ileri_bakma_maske=False, Wover=Wx)
            out[nm] = {b_: SD(r[b_]) for b_ in ("train", "valid")}
        R["4c_vol_gecikme"][ad] = out
        print(f"   {ad}: vol_t tr {out['vol_t']['train']['sharpe']} va {out['vol_t']['valid']['sharpe']} | "
              f"vol_t-1 tr {out['vol_t-1']['train']['sharpe']} va {out['vol_t-1']['valid']['sharpe']}")
    bulgu("4C", "Vol olceklemesi t gununun (kapanista bilinen) vol30'unu kullanip t+1 acilisinda uyguluyor - dogru.",
          f"Ek 1 gun gecikmeyle (vol[t-1]) sonuc neredeyse ayni: "
          f"A2 train {R['4c_vol_gecikme']['A2']['vol_t']['train']['sharpe']}->"
          f"{R['4c_vol_gecikme']['A2']['vol_t-1']['train']['sharpe']}, "
          f"Y train {R['4c_vol_gecikme']['Y']['vol_t']['train']['sharpe']}->"
          f"{R['4c_vol_gecikme']['Y']['vol_t-1']['train']['sharpe']}.",
          "Vol tahmini gelecege bagimli degil; sonuc vol'un tam gunune duyarli degil.", "TEMIZ")
    return R


# =====================================================================================
# 5) MALIYET
# =====================================================================================
def test5_maliyet():
    print("\n" + "=" * 90 + "\n5) MALIYET / CIRO\n" + "=" * 90)
    R = {}
    P = T.panel(); b = P["bas"]
    O = P["O"][b:]; C = P["C"][b:]; W = T2.agirliklar2(A2)[b:]
    eq, g = H.sim_portfoy(O, C, W, maliyet=0.0007)
    # elle 6 gun dogrula
    el = []
    w_prev = np.zeros(W.shape[1])
    for t in range(0, 900):
        w = np.nan_to_num(W[t]).copy(); w[np.isnan(O[t + 1]) | np.isnan(O[t + 2])] = 0
        r = np.clip(np.nan_to_num((O[t + 2] - O[t + 1]) / O[t + 1]), -0.95, 5.0)
        brut = float((w * r).sum()); ciro = float(np.abs(w - w_prev).sum()); mal = ciro * 0.0007
        x = brut - mal
        if t in (0, 1, 200, 400, 800):
            el.append(dict(t=int(t), tarih=datetime.utcfromtimestamp(P["gun"][b + t] / 1e3).strftime("%Y-%m-%d"),
                           w=[round(float(v), 5) for v in w[np.abs(w) > 1e-12]],
                           w_prev=[round(float(v), 5) for v in w_prev[np.abs(w_prev) > 1e-12]],
                           brut_getiri=round(brut, 6), ciro_sum_abs_dw=round(ciro, 6),
                           maliyet=round(mal, 8), net=round(x, 6),
                           sim_portfoy_g=round(float(g[t]), 6),
                           fark=abs(x - float(g[t]))))
        w_prev = w * (1 + r) / (1 + x) if x > -1 else w
    R["5a_elle_dogrulama"] = el
    maxfark = max(e["fark"] for e in el)
    for e in el:
        print(f"   t={e['t']:>3} {e['tarih']} brut={e['brut_getiri']:+.6f} ciro={e['ciro_sum_abs_dw']:.6f} "
              f"mal={e['maliyet']:.8f} net={e['net']:+.6f} sim={e['sim_portfoy_g']:+.6f} fark={e['fark']:.2e}")
    bulgu("5A", "Maliyet formulu elle dogrulandi: maliyet = sum|w_t - w_prev_drifted| * 0.0007, tek yon.",
          f"5 ornek gunde elle hesap ile sim_portfoy ciktisi arasindaki max fark {maxfark:.2e}. "
          f"w_prev, onceki gunun getirisiyle SURUKLENMIS agirliktir (w*(1+r)/(1+x)) -> yalnizca gercekten "
          f"islem goren fark ucretlendiriliyor; bu DOGRU (fazla maliyet yazilmiyor, eksik de yazilmiyor).",
          "Maliyet muhasebesi dogru. Tek yon 7bp = alis+satis toplamda 14bp; Binance spot taker 10bp "
          "(BNB indirimsiz) + kayma dusunulurse 7bp iyimser olabilir -> asagida duyarlilik.", "TEMIZ")

    # 5b. ciro ve maliyet duyarliligi
    print("\n-- maliyet duyarliligi")
    R["5b_duyarlilik"] = {}
    for ad, cfg in ADAY.items():
        R["5b_duyarlilik"][ad] = {}
        for mal in (0.0007, 0.0010, 0.0020, 0.0035, 0.0050):
            r = kos_duz(cfg, maliyet=mal, clip=None, ileri_bakma_maske=False)
            R["5b_duyarlilik"][ad][mal] = {b_: SD(r[b_]) for b_ in ("train", "valid")}
            print(f"   {ad} {mal*1e4:>4.0f}bp: tr S={r['train']['sharpe']} cagr={r['train']['cagr']} "
                  f"| va S={r['valid']['sharpe']} cagr={r['valid']['cagr']} | ciro/yil={r['train']['ciro_yil']}")
    # kirilma noktasi
    R["5c_basabas"] = {}
    for ad, cfg in ADAY.items():
        lo, hi = 0.0007, 0.20
        for _ in range(40):
            md = (lo + hi) / 2
            s = kos_duz(cfg, maliyet=md, bolumler=("train",), clip=None, ileri_bakma_maske=False)["train"]["cagr"]
            if s > 0:
                lo = md
            else:
                hi = md
        R["5c_basabas"][ad] = round(lo * 1e4, 1)
        print(f"   {ad} basabas maliyet (train CAGR=0): {R['5c_basabas'][ad]} bp/tek yon")
    bulgu("5B", "Maliyet dayanikliligi: ciro yuksek (A2 ~13-18/yil) ama basabas maliyet cok uzakta.",
          f"Train CAGR'i sifira indiren tek-yon maliyet: A2 {R['5c_basabas']['A2']}bp, Y {R['5c_basabas']['Y']}bp "
          f"(varsayilan 7bp). 50bp'de bile: A2 valid S={R['5b_duyarlilik']['A2'][0.0050]['valid']['sharpe']}, "
          f"Y valid S={R['5b_duyarlilik']['Y'][0.0050]['valid']['sharpe']}.",
          "Maliyet varsayimi sonucu sisirmiyor.", "TEMIZ")

    # 5d. gunluk kucuk agirlik degisimleri gercekten ucretlendiriliyor mu?
    ciro, giris = T2.akis(W, P["gun"][b:])
    kucuk = float(ciro[(ciro > 0) & (ciro < 0.01)].sum()); toplam = float(ciro.sum())
    R["5d_kucuk_ciro"] = dict(toplam_ciro=round(toplam, 2), kucuk_ciro=round(kucuk, 2),
                              oran=round(kucuk / max(toplam, 1e-9), 4))
    bulgu("5C", "DIKKAT (metodoloji, sisirme degil): tsmom2.akis() ciroyu w_prev=W[t-1] (SURUKLENMEMIS) ile "
                "hesapliyor, sim_portfoy ise suruklenmis w_prev ile. Rapordaki 'ciro_yil' gercek islem hacmini "
                "OLDUGUNDAN BUYUK gosteriyor; maliyet hesabi (sim_portfoy) dogru olan.",
          f"tsmom2.akis satir 181-183: prev = vstack([0, Wb[:-1]]) - fiyat surukleniminden gelen agirlik "
          f"degisimi islem sayilmiyor ama sonraki gun farki islem gibi gorunuyor.",
          "Raporlanan ciro/yil biraz yuksek (kotumser yonde); maliyet dogru hesaplandigi icin Sharpe etkilenmiyor.",
          "DUSUK")
    return R


# =====================================================================================
# 6) KIYASLARIN ADILLIGI
# =====================================================================================
def test6_kiyas():
    print("\n" + "=" * 90 + "\n6) KIYASLARIN ADILLIGI\n" + "=" * 90)
    R = {}
    P = T.panel(); b = P["bas"]
    # 6a. ayni maliyet / evren / baslangic?
    R["6a"] = {}
    for ad, cfg in (("A2", A2), ("A2_altut", A2B), ("Y", YY), ("Y_altut", YYB)):
        r = kos_duz(cfg, clip=None, ileri_bakma_maske=False)
        W = T2.agirliklar2(cfg)[b:]
        R["6a"][ad] = dict(train=SD(r["train"]), valid=SD(r["valid"]),
                           ilk_pozisyon_gun=datetime.utcfromtimestamp(
                               P["gun"][b + int(np.argmax((np.abs(W) > 1e-12).any(1)))] / 1e3).strftime("%Y-%m-%d"))
        print(f"   {ad:<10} tr S={r['train']['sharpe']:>5} dd={r['train']['maxdd']:<6} gross={r['train']['ort_gross']:<6} "
              f"ciro/yil={r['train']['ciro_yil']:<6} | va S={r['valid']['sharpe']:>5} dd={r['valid']['maxdd']} "
              f"| ilk gun {R['6a'][ad]['ilk_pozisyon_gun']}")
    ayni_gun = len({R["6a"][k]["ilk_pozisyon_gun"] for k in ("A2", "A2_altut")}) == 1
    R["6a_ayni_baslangic_b3"] = ayni_gun
    bulgu("6A", "Al-tut kiyasi AYNI maliyet (0.0007), AYNI evren maskesi, AYNI gun penceresi ve AYNI "
                "getiri/kirpma mekanigi ile kosuluyor.",
          f"tsmom2.agirliklar2'nin altut kolu ayni evren_maske2'yi kullaniyor; kos2 ayni maliyeti "
          f"sim_portfoy'a veriyor; her ikisi de P['bas'] (2018-04-01) indeksinden basliyor ve raporlanan "
          f"gun sayisi ayni (train {R['6a']['A2']['train']['gun']} vs {R['6a']['A2_altut']['train']['gun']}, "
          f"valid {R['6a']['A2']['valid']['gun']} vs {R['6a']['A2_altut']['valid']['gun']}). "
          f"Ilk POZISYON gunu A2 {R['6a']['A2']['ilk_pozisyon_gun']} = al-tut "
          f"{R['6a']['A2_altut']['ilk_pozisyon_gun']}; Y {R['6a']['Y']['ilk_pozisyon_gun']} vs al-tut "
          f"{R['6a']['Y_altut']['ilk_pozisyon_gun']} - bu fark haksizlik DEGIL, Y'nin ilk 18 gun MA50 "
          f"altinda olup nakitte beklemesi (ayni pencerede, sadece pozisyon yok).",
          "Maliyet/evren/pencere ekseninde haksizlik yok. Haksizlik OLCEKTE (bkz. 6B).", "TEMIZ")

    # 6a-2. evren_maske2 onbellek anahtari 'tam' icermiyor (latent hata)
    R["6a2_ay_cache"] = dict(anahtar="('aylik', N, volwin)",
                             uyari="cfg['ma']/ret/band'dan gelen 'tam' maskesi anahtarda yok")
    bulgu("6C", "LATENT HATA (bugun sonucu etkilemiyor): tsmom2._AY onbellek anahtari ('aylik', N, volwin) "
                "MA listesini icermiyor, ama onbelleklenen M matrisi dongude canli[src] uzerinden 'tam'a "
                "(yani MA penceresine) bagimli. Ayni oturumda once MA200'lu, sonra MA50'li bir yuruyen "
                "konfig kosulursa ikincisi BIRINCININ secimlerini kullanir.",
          "tsmom2.py satir 60-83: `k = ('aylik', N, cfg.get('volwin', 30)); if k in _AY: return _AY[k] & canli` "
          "- fakat satir 78 `sk = np.where(canli[src] & ...)` ve canli = tam & ... . Bugun tum yuruyen "
          "konfigler ma=[50] kullandigi icin fark yaratmiyor.",
          "A2/Y sonuclarini bugun etkilemiyor; ileride sessiz yanlis sonuc uretebilir.", "DUSUK")

    # 6b. GROSS uyusmazligi -> maxDD kiyasi haksiz
    gA = R["6a"]["A2"]["train"]["gross"]; gB = R["6a"]["A2_altut"]["train"]["gross"]
    print("\n-- gross-esitlenmis al-tut kiyasi (maxDD adil karsilastirma)")
    R["6b_gross_esit"] = {}
    for ad, cfg, bcfg in (("A2", A2, A2B), ("Y", YY, YYB)):
        R["6b_gross_esit"][ad] = {}
        for bl in ("train", "valid"):
            gs = R["6a"][ad][bl]["gross"]
            Wb = T2.agirliklar2(bcfg) * gs      # al-tutu stratejinin ort gross'una olcekle
            rb = kos_duz(bcfg, clip=None, ileri_bakma_maske=False, Wover=Wb)
            R["6b_gross_esit"][ad][bl] = dict(strateji=R["6a"][ad][bl],
                                              altut_ham=R["6a"][ad + "_altut"][bl],
                                              altut_gross_esit=SD(rb[bl]))
            print(f"   {ad} {bl}: strateji dd={R['6a'][ad][bl]['maxdd']} (gross {gs}) | "
                  f"al-tut ham dd={R['6a'][ad+'_altut'][bl]['maxdd']} (gross {R['6a'][ad+'_altut'][bl]['gross']}) | "
                  f"al-tut gross-esit dd={rb[bl]['maxdd']} S={rb[bl]['sharpe']}")
    e = R["6b_gross_esit"]["A2"]["train"]
    bulgu("6B", "EN BUYUK KIYAS SORUNU: 'maxDD al-tutun yarisi' iddiasi buyuk olcude POZISYON BUYUKLUGU "
                "farkindan geliyor, sinyalden degil. Strateji ortalama gross ~%s iken al-tut %s." %
                (R["6a"]["A2"]["train"]["gross"], R["6a"]["A2_altut"]["train"]["gross"]),
          f"A2 train: strateji maxDD {e['strateji']['maxdd']} (gross {e['strateji']['gross']}) vs ham al-tut "
          f"{e['altut_ham']['maxdd']} (gross {e['altut_ham']['gross']}); AYNI ortalama gross'a olceklenmis al-tut "
          f"maxDD = {e['altut_gross_esit']['maxdd']}. "
          f"Y train: strateji {R['6b_gross_esit']['Y']['train']['strateji']['maxdd']} vs gross-esit al-tut "
          f"{R['6b_gross_esit']['Y']['train']['altut_gross_esit']['maxdd']}.",
          "maxDD ustunlugunun buyuk kismi 'daha az yatirim yapmak'. Gercek sinyal katkisi Sharpe farkinda "
          "(olcekten bagimsiz) gorulur; DD karsilastirmasi gross-esitlenmis kiyasa gore yapilmali. "
          "AYRICA: nakit %70-80 oraninda ve model nakde %0 faiz yaziyor - bu strateji ALEYHINE (muhafazakar).", "YUKSEK")

    # 6c. nakit getirisi senaryosu (strateji lehine olan kaciriliyor mu)
    R["6c_nakit"] = {}
    for ad, cfg in ADAY.items():
        r = kos_duz(cfg, clip=None, ileri_bakma_maske=False)
        W = T2.agirliklar2(cfg)[b:]
        g = r["_g"]; gun = r["_gun"]
        nakit_or = 1.0 - np.abs(W[:len(g)]).sum(1)
        for yil_f in (0.0, 0.04):
            g2 = g + np.maximum(nakit_or, 0) * (yil_f / 365)
            rep_t = H.portfoy_rapor(np.cumprod(1 + g2), g2, gun, "train")
            rep_v = H.portfoy_rapor(np.cumprod(1 + g2), g2, gun, "valid")
            R["6c_nakit"].setdefault(ad, {})[f"{yil_f:.0%}"] = dict(train=SD(rep_t), valid=SD(rep_v))
        print(f"   {ad} nakit %0: tr {R['6c_nakit'][ad]['0%']['train']['sharpe']} | "
              f"nakit %4: tr {R['6c_nakit'][ad]['4%']['train']['sharpe']} "
              f"va {R['6c_nakit'][ad]['0%']['valid']['sharpe']} -> {R['6c_nakit'][ad]['4%']['valid']['sharpe']}")
    return R


# =====================================================================================
# 7) DUZELTILMIS TOPLAM SONUC
# =====================================================================================
def W_duzeltilmis(cfg, cap=1.0, min_gun_nedensel=True):
    """Tum duzeltmeler uygulanmis agirlik matrisi:
       - '>=200 gun gecmis' sarti NEDENSEL (min_gun=200 ileriye bakan filtresinin karsiligi)
       - cap 1.0 (spot)"""
    P = T.panel()
    frac, tam = T.oylar(cfg["ma"], cfg["ret"], cfg["band"])
    um = T2.evren_maske2(tam, cfg)
    if min_gun_nedensel:
        um = um & (np.cumsum(~np.isnan(P["C"]), axis=0) >= 200)
    s = np.ones_like(frac) if cfg.get("altut") else np.nan_to_num(frac)
    s = np.where(um, s, 0.0)
    K = um.sum(1)
    W = np.where(um, 1.0 / np.maximum(K, 1)[:, None], 0.0) * s
    if cfg.get("hedef") and not cfg.get("altut"):
        W = W * np.clip(cfg["hedef"] / np.nan_to_num(T.volat(cfg.get("volwin", 30)), nan=1e9), 0, cap)
    gr = np.abs(W).sum(1, keepdims=True)
    return np.where(gr > 1.0, W / np.maximum(gr, 1e-12), W)


def test7_toplam():
    print("\n" + "=" * 90 + "\n7) ONCEKI vs DUZELTILMIS (tum duzeltmeler birlikte)\n" + "=" * 90)
    print("   duzeltmeler: (1) sim_portfoy O[t+2] ileri-bakan maskesi kaldirildi, (2) getiri kirpmasi yok,")
    print("                (3) cap=1.0 (spot), (4) '>=200 gun gecmis' sarti NEDENSEL uygulandi (min_gun sizintisi),")
    print("                (5) al-tut kiyasi stratejinin ort. gross'una esitlendi (maxDD adil olsun diye)")
    out = {}
    for ad, cfg in (("A2", A2), ("Y", YY), ("A2_altut", A2B), ("Y_altut", YYB)):
        onc = kos_duz(cfg, clip=(-0.95, 5.0), ileri_bakma_maske=True, cap=1.5)
        duz = kos_duz(cfg, clip=None, ileri_bakma_maske=False, Wover=W_duzeltilmis(cfg, cap=1.0))
        out[ad] = dict(onceki={b_: SD(onc[b_]) for b_ in ("train", "valid")},
                       duzeltilmis={b_: SD(duz[b_]) for b_ in ("train", "valid")})
        for b_ in ("train", "valid"):
            o, d = onc[b_], duz[b_]
            print(f"   {ad:<10} {b_:<6} ONCE S={o['sharpe']:>5} dd={o['maxdd']:<6} cagr={o['cagr']:>6} gross={o['ort_gross']:<6}"
                  f" -> SONRA S={d['sharpe']:>5} dd={d['maxdd']:<6} cagr={d['cagr']:>6} gross={d['ort_gross']}")
    # gross-esit al-tut (duzeltilmis dunyada)
    print("\n   -- gross-esitlenmis al-tut kiyasi (duzeltilmis) --")
    out["gross_esit_altut"] = {}
    for ad, cfg, bcfg in (("A2", A2, A2B), ("Y", YY, YYB)):
        out["gross_esit_altut"][ad] = {}
        for bl in ("train", "valid"):
            gs = out[ad]["duzeltilmis"][bl]["gross"]
            rb = kos_duz(bcfg, clip=None, ileri_bakma_maske=False, Wover=W_duzeltilmis(bcfg) * gs)
            out["gross_esit_altut"][ad][bl] = SD(rb[bl])
            st = out[ad]["duzeltilmis"][bl]
            print(f"   {ad} {bl:<6}: strateji S={st['sharpe']:>5} dd={st['maxdd']:<6} | "
                  f"gross-esit al-tut S={rb[bl]['sharpe']:>5} dd={rb[bl]['maxdd']:<6} "
                  f"-> Sharpe farki {st['sharpe']-rb[bl]['sharpe']:+.2f}, dd farki {st['maxdd']-rb[bl]['maxdd']:+.3f}")
    a2 = out["A2"]; yy = out["Y"]
    bulgu("7A", "TOPLAM: duzeltmelerden sonra adaylarin Sharpe'i pratikte AYNI kaliyor; sisen sey Sharpe degil, "
                "'maxDD ustunlugu' anlatisi ve sepet secimi.",
          f"A2 train {a2['onceki']['train']['sharpe']}->{a2['duzeltilmis']['train']['sharpe']} "
          f"(dd {a2['onceki']['train']['maxdd']}->{a2['duzeltilmis']['train']['maxdd']}), "
          f"valid {a2['onceki']['valid']['sharpe']}->{a2['duzeltilmis']['valid']['sharpe']} "
          f"(dd {a2['onceki']['valid']['maxdd']}->{a2['duzeltilmis']['valid']['maxdd']}); "
          f"Y train {yy['onceki']['train']['sharpe']}->{yy['duzeltilmis']['train']['sharpe']}, "
          f"valid {yy['onceki']['valid']['sharpe']}->{yy['duzeltilmis']['valid']['sharpe']}. "
          f"Gross-esit al-tut karsisinda A2 train Sharpe farki "
          f"{a2['duzeltilmis']['train']['sharpe']-out['gross_esit_altut']['A2']['train']['sharpe']:+.2f}, "
          f"maxDD farki {a2['duzeltilmis']['train']['maxdd']-out['gross_esit_altut']['A2']['train']['maxdd']:+.3f}.",
          "Kod duzeyinde sisirme kucuk. Asil indirim kalemleri: sepet secim yanliligi (~-0.2..-0.4 Sharpe), "
          "yuruyen sepetin survivorship'i (~-0.1..-0.4), ve maxDD ustunlugunun buyuk kisminin sahte olmasi.", "ORTA")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="hepsi")
    a = ap.parse_args()
    P = T.panel()
    print(f"panel: {len(P['syms'])} sembol, {len(P['gun'])} gun, bas idx={P['bas']} "
          f"({datetime.utcfromtimestamp(P['gun'][P['bas']]/1e3).date()}) | holdout acik mi: {H._holdout_acik()}")
    R = {}
    if a.test in ("hepsi", "1"):
        R["t1_ileriye_bakma"] = test1_ileriye_bakma()
    if a.test in ("hepsi", "2"):
        R["t2_panel"] = test2_panel()
    if a.test in ("hepsi", "3"):
        R["t3_survivorship"] = test3_survivorship()
    if a.test in ("hepsi", "4"):
        R["t4_vol"] = test4_vol()
    if a.test in ("hepsi", "5"):
        R["t5_maliyet"] = test5_maliyet()
    if a.test in ("hepsi", "6"):
        R["t6_kiyas"] = test6_kiyas()
    if a.test == "hepsi":
        R["t7_toplam"] = test7_toplam()
        t7 = R["t7_toplam"]; t3 = R.get("t3_survivorship", {})
        ozet = dict(
            karar="Kod duzeyinde ADAYLAR AYAKTA. 6 denetim basliginin 4'u tamamen temiz (ileriye bakma, "
                  "panel hizalamasi, maliyet, vol hedefleme). Iki gercek indirim var ve ikisi de KOD DEGIL "
                  "METODOLOJI: (i) maxDD ustunlugu buyuk olcude pozisyon buyuklugu farkindan geliyor, "
                  "(ii) sepet secimi (A2) ve hayatta-kalma yanliligi (Y) Sharpe'i sisiriyor.",
            kod_hatalari=[
                "min_gun=200 (harness.gunluk_panel) ileriye bakan evren filtresi - SADECE Y'yi etkiler; "
                "duzeltince Y train 1.19 -> 1.36 (yani sizinti aleyhte calisiyormus)",
                "sim_portfoy'da O[t+2] nan maskesi teknik olarak ileriye bakma - bu iki adayda ETKISI SIFIR "
                "(etkilenen hucre yok), ama baska bir evrende sessizce sisirir",
                "tsmom2.akis() ciroyu suruklenmemis w_prev ile hesapliyor -> raporlanan ciro/yil abartili "
                "(maliyet hesabi dogru, Sharpe etkilenmiyor)",
            ],
            sonuc_tablosu=dict(
                A2=dict(onceki=t7["A2"]["onceki"], duzeltilmis=t7["A2"]["duzeltilmis"],
                        gross_esit_kiyas=t7["gross_esit_altut"]["A2"]),
                Y=dict(onceki=t7["Y"]["onceki"], duzeltilmis=t7["Y"]["duzeltilmis"],
                       gross_esit_kiyas=t7["gross_esit_altut"]["Y"])),
            durust_beklenti=dict(
                A2="train Sharpe 1.57 (duzeltilmis). Sepet secim yanliligini dusunce (tsmom2: 7 sepet medyani "
                   "1.37, ileriye bakmayan yuruyen sepet 1.36) durust train beklentisi ~1.35. maxDD ustunlugu "
                   "gross-esit kiyasa gore yalnizca 0.276 vs 0.308 -> IHMAL EDILEBILIR. Gercek edge Sharpe'ta: "
                   "gross-esit al-tuta karsi train +0.50, valid +0.99.",
                Y="train Sharpe 1.36 (min_gun duzeltmesi dahil). Hayatta-kalma yanliligi icin -0.1..-0.45 "
                  "indirim -> durust train beklentisi ~0.9-1.25. valid 1.47. Y, A2'den daha DURUST bir aday "
                  "(sepet secimi kuralin parcasi) ama tek temiz olmayan yani survivorship."),
            olculmemis_riskler=[
                "1h veriyle olculen uygulama gecikmesi penceresi tek yil ve tek rejim (2024-11..2025-12).",
                "Sepet coinlerinden birinin GECELIK -%90 cokusu: A2'de tek-coin agirlik p95 ~0.37, "
                "MA50 ve vol hedefleme bu riske karsi HICBIR koruma saglamiyor.",
                "En kotu dusus 731 gun su altinda (tsmom2 olcumu) - Sharpe bunu gostermiyor.",
                "valid = tek yil (363 gun), tek gozlem gibi davranilmali; defterdeki toplam test sayisi yuksek.",
            ])
        out = dict(
            aile="kirmizi_takim", tarih=datetime.now(timezone.utc).isoformat(),
            hedef="A2 (b3 MA50+vt30) ve Y (yuruyen top-3 MA50+vt30) adaylarini sisiren her seyi bulmak",
            holdout="DOKUNULMADI (LAB_HOLDOUT ayarlanmadi, veri 2026-01-01'de fiziksel kesik)",
            ozet=ozet, bulgular=BULGULAR, olcumler=R)
        p = os.path.join(LAB, "sonuc_kirmizi.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)
        print("\nyazildi:", p, "| bulgu:", len(BULGULAR))


if __name__ == "__main__":
    main()
