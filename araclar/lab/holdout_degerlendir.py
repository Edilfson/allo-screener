"""HOLDOUT (2026) DEGERLENDIRMESI - SADECE ana ajan, programin sonunda, TEK SEFER.
Calistirma: LAB_HOLDOUT=son-degerlendirme-2026 python holdout_degerlendir.py
Adaylar ONCEDEN ilan edildi (HIPOTEZLER.md / bu dosya); sonradan aday eklenmez.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import tsmom2 as T2

ADAYLAR = [
    # ad, sepet, cfg override, aciklama
    ("A1_b3_ma50",          "s3_orij_bnb",  dict(hedef=None), "BTC/ETH/BNB MA50 long-only (sepet geriye donuk secildi)"),
    ("A2_b3_ma50_vt30",     "s3_orij_bnb",  dict(hedef=0.3),  "A1 + hedef vol %30 cap 1.5 (ana aday)"),
    ("Y_yuruyen3_ma50_vt30","yuruyen_top3", dict(hedef=0.3),  "aylik gecmis-hacim top-3 + MA50 + vt30 (ileriye bakma yok)"),
    ("Y_yuruyen3_ma50",     "yuruyen_top3", dict(hedef=None), "yuruyen top-3 + MA50, vol hedefsiz"),
]
KIYAS = [
    ("K_b3_altut",       "s3_orij_bnb",  dict(altut=True), "BTC/ETH/BNB esit al-tut"),
    ("K_btc_altut",      "s1_btc",       dict(altut=True), "BTC al-tut"),
    ("K_yuruyen3_altut", "yuruyen_top3", dict(altut=True), "yuruyen top-3 al-tut"),
]

acik = H._holdout_acik()
bolumler = ("train", "valid", "holdout") if acik else ("train", "valid")
print("HOLDOUT ACIK" if acik else "holdout kapali (prova)", "| bolumler:", bolumler)
if "s1_btc" not in T2.SEPETLER:
    T2.SEPETLER["s1_btc"] = ["BTCUSDT"]
sonuc = {}
for ad, sepet, kw, acik_ad in ADAYLAR + KIYAS:
    cfg = T2.sepet_cfg(sepet, **kw); cfg["ad"] = ad
    r = T2.kos2(cfg, bolumler=bolumler, kaydet=acik, ad=ad, not_="HOLDOUT tek sefer" if acik else "prova")
    sonuc[ad] = {b: r[b] for b in bolumler}
    sonuc[ad]["aciklama"] = acik_ad
    print(f"\n{ad}  ({acik_ad})")
    for b in bolumler:
        x = r[b]
        if "sharpe" in x:
            print(f"  {b:<8} gun={x['gun']:>4} getiri={x['toplam']*100:+6.1f}% CAGR={x['cagr']*100:+6.1f}% "
                  f"maxDD={x['maxdd']*100:5.1f}% Sharpe={x['sharpe']:5.2f} ciro/yil={x.get('ciro_yil')}")
        else:
            print(f"  {b:<8} veri yok ({x})")
# BRUT-ESLENMIS KIYAS (kirmizi takim 6B): strateji ort brut ~0.26 ile kosuyor; al-tutu ayni brute
# olcekleyince maxDD farki buyuk olcude kayboluyor. Adil DD kiyasi icin A2 ve Y'nin her bolumdeki
# ortalama brutuyle olceklenmis al-tut.
print("\n== BRUT-ESLENMIS AL-TUT KIYASI ==")
P = T2.T.panel(); b = P["bas"]
for ad, sepet, kw, _ in ADAYLAR:
    if kw.get("hedef") is None:
        continue
    cfg = T2.sepet_cfg(sepet, **kw); cfg["ad"] = ad
    r = T2.kos2(cfg, bolumler=bolumler, kaydet=False)
    ka = T2.sepet_cfg(sepet, altut=True); ka["ad"] = "K_" + ad
    Wk = T2.agirliklar2(ka)
    for bl in bolumler:
        g = r[bl].get("ort_gross")
        if not g:
            continue
        eq, gg = H.sim_portfoy(P["O"][b:], P["C"][b:], Wk[b:] * g, maliyet=T2.MALIYET_VARSAYILAN)
        rk = H.portfoy_rapor(eq, gg, P["gun"][b:], bl)
        if "sharpe" in rk:
            print(f"  {ad:<24} {bl:<8} strateji Sharpe={r[bl]['sharpe']:.2f} maxDD={r[bl]['maxdd']*100:.1f}%  |  "
                  f"al-tut@brut{g:.2f} Sharpe={rk['sharpe']:.2f} maxDD={rk['maxdd']*100:.1f}%")
            sonuc.setdefault(ad, {})[f"kiyas_brut_esli_{bl}"] = rk
if acik:
    json.dump(sonuc, open(os.path.join(H.LAB, "sonuc_holdout.json"), "w"), indent=1, default=str)
    print("\nKaydedildi: sonuc_holdout.json")
