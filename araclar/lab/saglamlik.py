"""saglamlik - A2 ve Y adaylarinin ISTATISTIKSEL saglamlik testi (kaynak=1d, portfoy bazli).

Hayatta kalan adaylar:
  A2 = s3_orij_bnb (BTC/ETH/BNB) + MA50 + hedef vol %30, cap 1.5
  Y  = yuruyen_top3 (aylik 90g USDT hacmi ilk-3) + MA50 + hedef vol %30, cap 1.5

Bu aile YENI PARAMETRE ARAMAZ. tsmom2.kos2 / tsmom2.sepet_cfg import edilir; ayni kural
farkli istatistiksel testlerden gecirilir. SADECE 'gelistirme' bolumu (train+valid, <2026).
HOLDOUT'a dokunulmaz (LAB_HOLDOUT ayarlanmaz, ham .npy okunmaz).

Testler:
 1) Blok bootstrap (blok=30 gun, 2000 tekrar): strateji-altut Sharpe farki ve maxDD farki GA95/GA99
 2) Yil bazinda en iyi MA (20..200 adim 10); oracle / sabit MA50 / walk-forward(onceki 3 yil)
 3) Monte Carlo: ay bloklari karistirilarak (10000) maxDD ve su-alti suresi dagilimi
 4) Deflated Sharpe (Bailey & Lopez de Prado 2014) - defterdeki tsmom+tsmom2 test sayisi ile
 5) Alt donemler: 2018-2020 / 2021-2022 / 2023-2025 strateji vs al-tut
 6) Uygulanabilirlik: vol hedefi %20 ve carpan tavani 1.0 (spot, kaldiracsiz)

Kullanim:
  PYTHONIOENCODING=utf-8 python saglamlik.py --faz hepsi
  PYTHONIOENCODING=utf-8 python saglamlik.py --faz bootstrap|yil|montecarlo|dsr|altdonem|kaldirac
"""
import argparse, json, os, sys, math
from datetime import datetime, timezone
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import tsmom as T
import tsmom2 as T2

LAB = os.path.dirname(os.path.abspath(__file__))
AILE = "saglamlik"
BOLUM = "gelistirme"                       # train+valid, <2026. HOLDOUT'a dokunulmaz.
RNG = np.random.default_rng(20260913)

ADAYLAR = {                                 # ad -> (sepet, ek cfg)
    "A2": ("s3_orij_bnb", dict(hedef=0.30)),
    "Y":  ("yuruyen_top3", dict(hedef=0.30)),
}
MA_IZGARA = list(range(20, 201, 10))
YILLAR = list(range(2018, 2026))
ALT_DONEMLER = [("2018_2020", 2018, 2020), ("2021_2022", 2021, 2022), ("2023_2025", 2023, 2025)]


# ------------------------------------------------------------------ seri yardimcilari
_SERI = {}


def seri(sepet, **kw):
    """gelistirme bolumune kesilmis (gun, gunluk getiri). tsmom2.kos2 kullanir (yeni kod yok)."""
    k = (sepet, json.dumps(kw, sort_keys=True, default=str))
    if k in _SERI:
        return _SERI[k]
    cfg = T2.sepet_cfg(sepet, **kw)
    r = T2.kos2(cfg, (), kaydet=False)                     # rapor/defter yazmadan sadece seri
    g = np.asarray(r["_g"]); gun = np.asarray(r["_gun"])[:len(g)]
    a, b = H.BOLUM[BOLUM]
    m = (gun >= a) & (gun < b)
    _SERI[k] = (gun[m], g[m])
    return _SERI[k]


def _isit():
    """DETERMINIZM: tsmom2._AY onbellegi, yuruyen sepetin aylik secimini ILK cagrilan cfg'nin
    'tam' maskesiyle (yani ilk MA uzunlugunun isinma penceresiyle) dondurur. Tum fazlarda ayni
    evren kullanilsin diye onbellegi HER ZAMAN referans kural (MA50 + vt30) ile isitiyoruz."""
    for sp, kw in ADAYLAR.values():
        seri(sp, **kw)


def yil_of(gun):
    return np.array([datetime.fromtimestamp(x / 1e3, timezone.utc).year for x in gun])


def ay_of(gun):
    return np.array([(lambda d: d.year * 12 + d.month)(datetime.fromtimestamp(x / 1e3, timezone.utc)) for x in gun])


def sharpe(g):
    g = np.asarray(g, float)
    s = g.std()
    return float(g.mean() / s * math.sqrt(365)) if s > 0 and len(g) > 2 else 0.0


def maxdd(g):
    eq = np.cumprod(1 + np.asarray(g, float))
    pk = np.maximum.accumulate(eq)
    return float(((pk - eq) / pk).max())


def su_alti(g):
    """en uzun kesintisiz 'onceki tepenin altinda' gun sayisi"""
    eq = np.cumprod(1 + np.asarray(g, float))
    pk = np.maximum.accumulate(eq)
    u = eq < pk
    best = cur = 0
    for x in u:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


def ozet(g, gun=None):
    g = np.asarray(g, float)
    eq = np.cumprod(1 + g); yil = len(g) / 365
    return dict(gun=int(len(g)), toplam=round(float(eq[-1] - 1), 3),
                cagr=round(float(eq[-1] ** (1 / max(yil, 1e-9)) - 1), 3),
                maxdd=round(maxdd(g), 3), sharpe=round(sharpe(g), 2),
                su_alti_gun=su_alti(g))


# ------------------------------------------------------------------ 1) blok bootstrap
def _mat_sharpe(X):
    s = X.std(1)
    return np.where(s > 0, X.mean(1) / s * math.sqrt(365), 0.0)


def _mat_maxdd(X):
    eq = np.cumprod(1 + X, axis=1)
    pk = np.maximum.accumulate(eq, axis=1)
    return ((pk - eq) / pk).max(1)


def blok_bootstrap(gs, gb, blok=30, B=2000, parca=250):
    """ESLESMIS hareketli blok bootstrap: ayni blok indeksleri hem stratejiye hem al-tuta uygulanir.
    Return: fark dagilimlarinin GA'lari + gozlenen farklar."""
    n = len(gs); nb = int(math.ceil(n / blok))
    d_sh = np.empty(B); d_dd = np.empty(B)
    ofs = np.arange(blok)
    for a in range(0, B, parca):
        b = min(a + parca, B)
        st = RNG.integers(0, n - blok + 1, size=(b - a, nb))
        idx = (st[:, :, None] + ofs[None, None, :]).reshape(b - a, nb * blok)[:, :n]
        Xs = gs[idx]; Xb = gb[idx]
        d_sh[a:b] = _mat_sharpe(Xs) - _mat_sharpe(Xb)
        d_dd[a:b] = _mat_maxdd(Xs) - _mat_maxdd(Xb)
    q = lambda x, p: round(float(np.quantile(x, p)), 3)
    return dict(
        B=B, blok_gun=blok, n_gun=int(n),
        gozlenen=dict(sharpe_strateji=round(sharpe(gs), 2), sharpe_altut=round(sharpe(gb), 2),
                      sharpe_fark=round(sharpe(gs) - sharpe(gb), 3),
                      maxdd_strateji=round(maxdd(gs), 3), maxdd_altut=round(maxdd(gb), 3),
                      maxdd_fark=round(maxdd(gs) - maxdd(gb), 3)),
        sharpe_fark=dict(ort=q(d_sh, .5), ga95=[q(d_sh, .025), q(d_sh, .975)],
                         ga99=[q(d_sh, .005), q(d_sh, .995)],
                         p_fark_sifir_veya_alti=round(float((d_sh <= 0).mean()), 4),
                         sifiri_disliyor_95=bool(np.quantile(d_sh, .025) > 0),
                         sifiri_disliyor_99=bool(np.quantile(d_sh, .005) > 0)),
        maxdd_fark=dict(ort=q(d_dd, .5), ga95=[q(d_dd, .025), q(d_dd, .975)],
                        ga99=[q(d_dd, .005), q(d_dd, .995)],
                        p_fark_sifir_veya_ustu=round(float((d_dd >= 0).mean()), 4),
                        sifiri_disliyor_95=bool(np.quantile(d_dd, .975) < 0),
                        sifiri_disliyor_99=bool(np.quantile(d_dd, .995) < 0)))


def faz_bootstrap(kaydet=True):
    out = {}
    for ad, (sp, kw) in ADAYLAR.items():
        gun, gs = seri(sp, **kw)
        gun2, gb = seri(sp, altut=True)
        assert len(gs) == len(gb) and (gun == gun2).all()
        r = blok_bootstrap(gs, gb)
        out[ad] = dict(sepet=sp, **r)
        if kaydet:
            H.kaydet(AILE, f"{ad}_bootstrap", dict(sepet=sp, kural="ma50_vt30", blok=30, B=2000),
                     BOLUM, r, "blok bootstrap: strateji - altut farki")
        print(f"{ad:<3} {sp:<14} Sharpe fark {r['gozlenen']['sharpe_fark']:>6} "
              f"GA95={r['sharpe_fark']['ga95']} GA99={r['sharpe_fark']['ga99']} p={r['sharpe_fark']['p_fark_sifir_veya_alti']} | "
              f"maxDD fark {r['gozlenen']['maxdd_fark']:>6} GA95={r['maxdd_fark']['ga95']}")
    return out


# ------------------------------------------------------------------ 2) yil bazinda MA kararliligi
def faz_yil(kaydet=True):
    out = {}
    for ad, (sp, kw) in ADAYLAR.items():
        seriler = {}
        for p in MA_IZGARA:
            gun, g = seri(sp, ma=[p], **kw)
            seriler[p] = (gun, g)
            if kaydet:
                H.kaydet(AILE, f"{ad}_ma{p}_vt30", dict(sepet=sp, ma=[p], hedef=0.30), BOLUM,
                         ozet(g), "yil-bazinda MA taramasi")
        gun = seriler[MA_IZGARA[0]][0]
        yl = yil_of(gun)
        # her yil x her MA sharpe
        tab = {}
        for y in YILLAR:
            m = yl == y
            if m.sum() < 60:
                continue
            tab[y] = {p: round(sharpe(seriler[p][1][m]), 2) for p in MA_IZGARA}
        en_iyi = {y: max(tab[y], key=lambda p: tab[y][p]) for y in tab}
        # 'iyi bolge' genisligi: en iyinin %90'i ustundeki MA sayisi
        bolge = {}
        for y in tab:
            v = tab[y]; mx = v[en_iyi[y]]
            iyi = [p for p in MA_IZGARA if v[p] >= mx - 0.20]
            bolge[y] = [min(iyi), max(iyi), len(iyi)]
        # --- 3 politika: oracle / sabit MA50 / walk-forward (onceki 3 yil)
        wf_sec = {}
        for y in sorted(tab):
            gec = [yy for yy in sorted(tab) if y - 3 <= yy < y]
            if len(gec) < 3:
                continue
            mg = np.isin(yl, gec)
            wf_sec[y] = max(MA_IZGARA, key=lambda p: sharpe(seriler[p][1][mg]))
        wf_yillar = sorted(wf_sec)
        pol = {}
        for etiket, secici in (("oracle_yil", lambda y: en_iyi[y]),
                               ("sabit_ma50", lambda y: 50),
                               ("walk_forward", lambda y: wf_sec[y])):
            for pencere, ys in (("2021_2025", wf_yillar), ("tum_2018_2025", sorted(tab))):
                if etiket == "walk_forward" and pencere != "2021_2025":
                    continue
                parcalar = [seriler[secici(y)][1][yl == y] for y in ys]
                gg = np.concatenate(parcalar)
                pol.setdefault(pencere, {})[etiket] = dict(
                    sharpe=round(sharpe(gg), 2), maxdd=round(maxdd(gg), 3),
                    cagr=round(float(np.prod(1 + gg) ** (365 / len(gg)) - 1), 3),
                    secimler={int(y): int(secici(y)) for y in ys})
        # altut kiyasi ayni pencerede
        _, gb = seri(sp, altut=True)
        for pencere, ys in (("2021_2025", wf_yillar), ("tum_2018_2025", sorted(tab))):
            m = np.isin(yl, ys)
            pol[pencere]["altut"] = dict(sharpe=round(sharpe(gb[m]), 2), maxdd=round(maxdd(gb[m]), 3))
        sec = [en_iyi[y] for y in sorted(en_iyi)]
        out[ad] = dict(sepet=sp, yil_x_ma_sharpe=tab, yilin_en_iyi_ma=en_iyi,
                       en_iyi_ma_std=round(float(np.std(sec)), 1),
                       en_iyi_ma_araligi=[int(min(sec)), int(max(sec))],
                       iyi_bolge_min_max_adet=bolge, politikalar=pol,
                       walk_forward_secimleri=wf_sec)
        if kaydet:
            H.kaydet(AILE, f"{ad}_yil_ma_kararlilik", dict(sepet=sp, izgara=[20, 200, 10]), BOLUM,
                     dict(yilin_en_iyi_ma=en_iyi, politikalar={k: {kk: vv.get("sharpe") for kk, vv in v.items()}
                                                               for k, v in pol.items()}),
                     "yil bazinda MA kararliligi")
        print(f"{ad} yilin en iyi MA: {en_iyi} | std={out[ad]['en_iyi_ma_std']}")
        for pencere in pol:
            print("   ", pencere, {k: v.get("sharpe") for k, v in pol[pencere].items()})
    return out


# ------------------------------------------------------------------ 3) Monte Carlo (ay bloklari)
def monte_carlo(g, gun, B=10000, tohum=7):
    ay = ay_of(gun)
    kesim = np.flatnonzero(np.diff(ay)) + 1
    bloklar = np.split(np.asarray(g, float), kesim)
    rng = np.random.default_rng(tohum)
    K = len(bloklar)
    dd = np.empty(B); ua = np.empty(B, int)
    for i in range(B):
        x = np.concatenate([bloklar[j] for j in rng.permutation(K)])
        eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
        dd[i] = ((pk - eq) / pk).max()
        u = eq < pk
        # en uzun su alti serisi (vektorel)
        if u.any():
            idx = np.flatnonzero(np.diff(np.concatenate([[0], u.view(np.int8), [0]])))
            ua[i] = int((idx[1::2] - idx[0::2]).max())
        else:
            ua[i] = 0
    q = lambda x, p: round(float(np.quantile(x, p)), 3)
    return dict(B=B, ay_blok_sayisi=int(K),
                gozlenen=dict(maxdd=round(maxdd(g), 3), su_alti_gun=su_alti(g)),
                maxdd_dagilim=dict(medyan=q(dd, .5), p75=q(dd, .75), p90=q(dd, .90),
                                   p95=q(dd, .95), p99=q(dd, .99), en_kotu=q(dd, 1.0),
                                   gozlenenin_yuzdeligi=round(float((dd <= maxdd(g)).mean()), 3)),
                su_alti_dagilim=dict(medyan=int(np.quantile(ua, .5)), p90=int(np.quantile(ua, .90)),
                                     p95=int(np.quantile(ua, .95)), p99=int(np.quantile(ua, .99)),
                                     en_kotu=int(ua.max()),
                                     gozlenenin_yuzdeligi=round(float((ua <= su_alti(g)).mean()), 3)))


def faz_montecarlo(kaydet=True, B=10000):
    out = {}
    for ad, (sp, kw) in ADAYLAR.items():
        gun, g = seri(sp, **kw)
        r = monte_carlo(g, gun, B=B)
        out[ad] = dict(sepet=sp, **r)
        if kaydet:
            H.kaydet(AILE, f"{ad}_montecarlo", dict(sepet=sp, kural="ma50_vt30", blok="ay", B=B),
                     BOLUM, r, "ay-bloklu karistirma maxDD/su-alti dagilimi")
        print(f"{ad} MC maxDD: gozlenen {r['gozlenen']['maxdd']} | medyan {r['maxdd_dagilim']['medyan']} "
              f"p95 {r['maxdd_dagilim']['p95']} p99 {r['maxdd_dagilim']['p99']} | "
              f"su alti gozlenen {r['gozlenen']['su_alti_gun']}g p95 {r['su_alti_dagilim']['p95']}g")
    return out


# ------------------------------------------------------------------ 4) Deflated Sharpe
def defter_denemeler(aileler=("tsmom", "tsmom2"), bolum="train"):
    """Defterdeki BAGIMSIZ konfigurasyon sayisi ve train Sharpe dagilimi (kiyas/al-tut haric)."""
    d = H.defter_oku()
    tr = {}
    for k in d:
        if k.get("aile") not in aileler or k.get("bolum") != bolum:
            continue
        r = k.get("rapor") or {}
        if not isinstance(r, dict) or r.get("sharpe") is None:
            continue
        ad = k["ad"]
        key = (k["aile"], ad, json.dumps(k.get("params"), sort_keys=True, default=str))
        tr[key] = float(r["sharpe"])
    kiyas = {k: v for k, v in tr.items() if k[1].startswith("KIYAS") or "altut" in k[1]}
    strat = {k: v for k, v in tr.items() if k not in kiyas}
    v = np.array(list(strat.values()))
    return dict(N=len(strat), kiyas=len(kiyas), sharpe_ort=round(float(v.mean()), 3),
                sharpe_std=round(float(v.std(ddof=1)), 3), sharpe_max=round(float(v.max()), 3),
                sharpe_min=round(float(v.min()), 3)), strat


def deflated_sharpe(g, N_test, sr_std_yillik, etiket=""):
    """Bailey & Lopez de Prado (2014). Tum hesap GUNLUK frekansta yapilir.
       SR0 = beklenen en yuksek Sharpe (N bagimsiz denemede, sifir gercek kenar altinda)
       DSR = P(gercek SR > SR0)"""
    g = np.asarray(g, float)
    n = len(g)
    sr = float(g.mean() / g.std())                       # gunluk, yillanmamis
    sk = float(stats.skew(g)); ku = float(stats.kurtosis(g, fisher=False))
    s_sr = sr_std_yillik / math.sqrt(365)                # denemeler arasi Sharpe std, gunluk
    eul = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / N_test)
    z2 = stats.norm.ppf(1 - 1.0 / (N_test * math.e))
    sr0 = s_sr * ((1 - eul) * z1 + eul * z2)
    payda = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr * sr, 1e-12))
    z = (sr - sr0) * math.sqrt(n - 1) / payda
    dsr = float(stats.norm.cdf(z))
    psr0 = float(stats.norm.cdf(sr * math.sqrt(n - 1) / payda))       # PSR vs SR=0
    # esik: DSR=0.95 icin gereken yillik Sharpe
    gerek = (sr0 + stats.norm.ppf(0.95) * payda / math.sqrt(n - 1)) * math.sqrt(365)
    return dict(etiket=etiket, gun=n, sharpe_yillik=round(sr * math.sqrt(365), 3),
                sharpe_gunluk=round(sr, 5), carpiklik=round(sk, 3), basiklik=round(ku, 2),
                N_test=N_test, denemeler_sharpe_std_yillik=round(sr_std_yillik, 3),
                SR0_yillik=round(sr0 * math.sqrt(365), 3), SR0_gunluk=round(sr0, 5),
                z=round(z, 3), deflated_sharpe=round(dsr, 4), psr_vs_sifir=round(psr0, 4),
                dsr95_icin_gereken_yillik_sharpe=round(gerek, 3))


def faz_dsr(kaydet=True):
    ist, _ = defter_denemeler()
    out = dict(defter=ist, hesaplar={})
    for ad, (sp, kw) in ADAYLAR.items():
        gun, g = seri(sp, **kw)
        a, b = H.BOLUM["train"]
        m = (gun >= a) & (gun < b)
        for bl, gg in (("train", g[m]), ("gelistirme", g)):
            for N in (ist["N"], 50, 400):
                r = deflated_sharpe(gg, N, ist["sharpe_std"], f"{ad}_{bl}_N{N}")
                out["hesaplar"][f"{ad}_{bl}_N{N}"] = r
                if kaydet:
                    H.kaydet(AILE, f"{ad}_dsr_{bl}_N{N}", dict(sepet=sp, kural="ma50_vt30", N_test=N),
                             bl, r, "deflated sharpe (Bailey-Lopez de Prado)")
            h = out["hesaplar"]["%s_%s_N%d" % (ad, bl, ist["N"])]
            print(f"{ad} {bl:<11} SR={h['sharpe_yillik']:>6} SR0(N={ist['N']})={h['SR0_yillik']:>6} "
                  f"DSR={h['deflated_sharpe']} PSR={h['psr_vs_sifir']} "
                  f"carpiklik={h['carpiklik']} basiklik={h['basiklik']}")
    return out


# ------------------------------------------------------------------ 5) alt donemler
def faz_altdonem(kaydet=True):
    out = {}
    for ad, (sp, kw) in ADAYLAR.items():
        gun, gs = seri(sp, **kw)
        _, g1 = seri(sp)                          # vol hedeflemesiz MA50 (referans)
        _, gb = seri(sp, altut=True)
        yl = yil_of(gun)
        d = {}
        for etiket, y0, y1 in ALT_DONEMLER:
            m = (yl >= y0) & (yl <= y1)
            rs, r1, rb = ozet(gs[m]), ozet(g1[m]), ozet(gb[m])
            d[etiket] = dict(vt30=rs, ma50=r1, altut=rb,
                             sharpe_fark=round(rs["sharpe"] - rb["sharpe"], 2),
                             maxdd_fark=round(rs["maxdd"] - rb["maxdd"], 3))
            if kaydet:
                H.kaydet(AILE, f"{ad}_altdonem_{etiket}", dict(sepet=sp, kural="ma50_vt30", donem=[y0, y1]),
                         BOLUM, d[etiket], "alt donem")
            print(f"{ad} {etiket}: vt30 S={rs['sharpe']} dd={rs['maxdd']} | ma50 S={r1['sharpe']} dd={r1['maxdd']} "
                  f"| altut S={rb['sharpe']} dd={rb['maxdd']}")
        out[ad] = dict(sepet=sp, donemler=d,
                       kac_donemde_sharpe_ustun=sum(1 for k in d if d[k]["sharpe_fark"] > 0),
                       kac_donemde_dd_dusuk=sum(1 for k in d if d[k]["maxdd_fark"] < 0))
    return out


# ------------------------------------------------------------------ 6) kaldirac / uygulanabilirlik
KALDIRAC = [("vt30_cap15_temel", dict(hedef=0.30, cap=1.5)),
            ("vt30_cap10_spot", dict(hedef=0.30, cap=1.0)),
            ("vt20_cap15", dict(hedef=0.20, cap=1.5)),
            ("vt20_cap10_spot", dict(hedef=0.20, cap=1.0)),
            ("ma50_vt_yok", dict())]


def faz_kaldirac(kaydet=True):
    out = {}
    for ad, (sp, _kw) in ADAYLAR.items():
        out[ad] = dict(sepet=sp, varyantlar={})
        for etiket, kw in KALDIRAC:
            cfg = T2.sepet_cfg(sp, **kw)
            r = T2.kos2(cfg, (BOLUM,), kaydet=False)
            rap = {k: v for k, v in r[BOLUM].items() if not k.startswith("_")}
            gun = np.asarray(r["_gun"])[:len(r["_g"])]
            g = np.asarray(r["_g"])
            m = (gun >= H.BOLUM[BOLUM][0]) & (gun < H.BOLUM[BOLUM][1])
            rap["su_alti_gun"] = su_alti(g[m])
            W = r["_W"][:len(g)][m]
            rap["maks_gross"] = round(float(np.abs(W).sum(1).max()), 3)
            out[ad]["varyantlar"][etiket] = rap
            if kaydet:
                H.kaydet(AILE, f"{ad}_{etiket}", dict(sepet=sp, **kw), BOLUM, rap, "kaldirac/uygulanabilirlik")
            print(f"{ad} {etiket:<18} S={rap.get('sharpe'):>5} cagr={rap.get('cagr'):>6} dd={rap.get('maxdd'):<6} "
                  f"gross ort={rap.get('ort_gross')} maks={rap['maks_gross']} ciro/yil={rap.get('ciro_yil')}")
        _, gb = seri(sp, altut=True)
        out[ad]["altut"] = ozet(gb)
    return out


# ------------------------------------------------------------------ rapor
def yaz(res):
    d = [k for k in H.defter_oku() if k.get("aile") == AILE]
    n = len({(k["ad"], k["bolum"], json.dumps(k["params"], default=str, sort_keys=True)) for k in d})
    bs, yil, mc, dsr, alt, kal = (res[k] for k in ("bootstrap", "yil", "montecarlo", "dsr", "altdonem", "kaldirac"))

    def karar_metni():
        p = []
        N = dsr["defter"]["N"]
        for ad in ("A2", "Y"):
            b = bs[ad]; h = dsr["hesaplar"]["%s_train_N%d" % (ad, N)]
            p.append("%s: Sharpe farki %+.2f (GA95 %s, GA99 %s, p=%s), maxDD farki %+.3f (GA95 %s); "
                     "DSR(N=%d, train)=%s" % (
                         ad, b["gozlenen"]["sharpe_fark"], b["sharpe_fark"]["ga95"], b["sharpe_fark"]["ga99"],
                         b["sharpe_fark"]["p_fark_sifir_veya_alti"], b["gozlenen"]["maxdd_fark"],
                         b["maxdd_fark"]["ga95"], N, h["deflated_sharpe"]))
        return " | ".join(p)

    # sablon uyumu: bu aile secim yapmaz; "en_iyi_3" = test edilen 2 aday + onerilen spot varyant
    en_iyi = []
    for ad, etiket in (("A2", "vt30_cap15_temel"), ("Y", "vt30_cap15_temel"), ("A2", "vt20_cap10_spot")):
        sp = ADAYLAR[ad][0]
        en_iyi.append(dict(
            ad=f"{ad}__{etiket}", params=dict(sepet=sp, kural="MA50 long-only", **dict(KALDIRAC)[etiket]),
            gelistirme=kal[ad]["varyantlar"][etiket], altut_gelistirme=kal[ad]["altut"],
            train_valid="bkz. sonuc_tsmom.json / sonuc_tsmom2.json (bu ailede yeniden secim yapilmadi)",
            komsular=("MA 20..200 taramasi: sabit MA50 tum-donem Sharpe %s, mukemmel yillik secim %s, "
                      "walk-forward %s -> komsu bolge duz, tepe gurultu." % (
                          yil[ad]["politikalar"]["tum_2018_2025"]["sabit_ma50"]["sharpe"],
                          yil[ad]["politikalar"]["tum_2018_2025"]["oracle_yil"]["sharpe"],
                          yil[ad]["politikalar"]["2021_2025"]["walk_forward"]["sharpe"])),
            saglamlik=dict(
                bootstrap_sharpe_fark_ga95=bs[ad]["sharpe_fark"]["ga95"],
                bootstrap_sharpe_fark_ga99=bs[ad]["sharpe_fark"]["ga99"],
                bootstrap_maxdd_fark_ga99=bs[ad]["maxdd_fark"]["ga99"],
                deflated_sharpe_train=dsr["hesaplar"]["%s_train_N%d" % (ad, dsr["defter"]["N"])]["deflated_sharpe"],
                mc_maxdd_p95=mc[ad]["maxdd_dagilim"]["p95"], mc_su_alti_p95=mc[ad]["su_alti_dagilim"]["p95"])))
    out = dict(
        aile=AILE, denenen=n, kaynak="1d", bolum="SADECE gelistirme (train+valid, <2026). HOLDOUT'a dokunulmadi.",
        en_iyi_3=en_iyi,
        analiz_baslangici="2018-04-01 (tsmom/tsmom2 ile ayni pencere)",
        adaylar={k: dict(sepet=v[0], kural="MA50 long-only, hedef vol %30, cap 1.5, gunluk rebalans, 7bp")
                 for k, v in ADAYLAR.items()},
        amac="A2 ve Y'nin sonuclari sanstan ayirt edilebilir mi? Yeni parametre aramasi YOK; "
             "tsmom2.kos2/sepet_cfg import edilerek ayni kural test edildi.",
        t1_blok_bootstrap=bs, t2_yil_ma_kararliligi=yil, t3_monte_carlo=mc,
        t4_deflated_sharpe=dsr, t5_alt_donemler=alt, t6_kaldirac=kal,
        cevaplar=CEVAPLAR, karar=KARAR, karar_ozet_sayilar=karar_metni(),
        dusenler=DUSENLER, ogrenilen=OGRENILEN, oneri=ONERI, notlar=NOTLAR)
    p = os.path.join(LAB, "sonuc_saglamlik.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nyazildi:", p, "| denenen:", n)
    return out, p


# ------------------------------------------------------------------ yorum metinleri (kosu sonrasi dolduruldu)
CEVAPLAR = {
    "S1_blok_bootstrap":
        "AYRIM MAXDD'DE NET, SHARPE'TA SINIRDA. Eslesmis hareketli blok bootstrap (blok=30 gun, 2000 tekrar, "
        "n=2830 gun; ayni blok indeksleri hem stratejiye hem KENDI sepetinin al-tutuna uygulandi). "
        "A2: Sharpe farki +0.58, GA95 [+0.06, +1.19] -> sifiri DISLIYOR; GA99 [-0.12, +1.33] -> sifiri "
        "DISLAMIYOR; p(fark<=0)=0.018. Y: +0.64, GA95 [+0.12, +1.21] disliyor, GA99 [-0.03, +1.38] "
        "dislamiyor, p=0.009. "
        "maxDD farki A2 -0.497 GA95 [-0.713,-0.323] GA99 [-0.759,-0.269]; Y -0.560 GA95 [-0.750,-0.396] "
        "GA99 [-0.780,-0.347]; her ikisinde de 2000 tekrarin 2000'unde fark negatif (p=0.000). "
        "YORUM: 'bu kural al-tuttan daha DUSUK dususle calisiyor' iddiasi %99'da ayakta; 'daha yuksek "
        "risk-ayarli GETIRI uretiyor' iddiasi %95'te ayakta, %99'da degil. GA'lar cok genis (Sharpe farki "
        "icin ~1.3 birim) - 8 yil bu soruyu kesin cevaplamak icin KISA.",
    "S2_yil_bazinda_parametre":
        "YILIN EN IYI MA'SI ZIPLIYOR AMA BU GURULTU; SABIT MA50 DOGRU SECIM. "
        "A2 yilin en iyisi: 2018:140, 2019:90, 2020:30, 2021:40, 2022:20, 2023:50, 2024:110, 2025:50 "
        "(std 40 gun, aralik 20-140). Y: 200/90/30/40/30/60/20/40 (std 55). Kararli bir bolge YOK: "
        "boga yillarinda kisa (20-50), 2018 ve 2024 gibi yatay/testere yillarinda uzun (110-200) MA kazaniyor. "
        "AMA bu ziplama ise yaramiyor: 2021-2025 penceresinde Sharpe - oracle (her yil o yilin en iyisi, "
        "ILERIYE BAKAR) A2 1.59 / Y 1.29; sabit MA50 A2 1.48 / Y 1.08; walk-forward (onceki 3 yilin en iyisi) "
        "A2 1.05 / Y 0.66; al-tut A2 0.98 / Y 0.55. "
        "Uc sonuc: (a) MUKEMMEL ongoruyle bile kazanc sadece +0.11/+0.21 Sharpe -> MA50 optimumun cok "
        "yakininda, parametre secimi bu ailede buyuk bir serbestlik derecesi DEGIL; (b) walk-forward sabit "
        "MA50'nin 0.43/0.42 ALTINDA -> yilin en iyi MA'sinin bir sonraki yila TASINMA gucu SIFIR, hatta "
        "negatif (secim tam tersi rejime uyum sagliyor); (c) tum 2018-2025'te oracle 1.83 vs sabit 1.57 = "
        "+0.26 -> 8 serbest secimin urettigi asiri-uydurma primi ~0.26 Sharpe; bu, bir parametre basina "
        "beklenen sisme buyuklugu icin somut bir olcek.",
    "S3_monte_carlo":
        "GOZLENEN DUSUS SANSLI DEGIL; %95 KOTU SENARYO ~%35 VE ~2.5-3 YIL SU ALTI. "
        "Gunluk getiriler 93 takvim-ayi blogu halinde 10000 kez karistirildi (blok ici sira korundu, "
        "bloklar arasi permutasyon). A2: gozlenen maxDD %27.5 dagilimin %75'lik diliminde (medyan %24.1, "
        "p90 %31.5, p95 %34.1, p99 %39.3, en kotu %53.2). Y: gozlenen %27.3, %72'lik dilim (medyan %24.2, "
        "p95 %34.7, p99 %39.3). Yani tarihsel maxDD sansli bir siralamadan gelmiyor - ortalamanin biraz "
        "ustunde, tipik. "
        "SU ALTI SURESI daha vahim: A2 gozlenen 730 gun dagilimin %89'unda (medyan 454, p90 745, p95 857, "
        "p99 1103, en kotu 1559 gun); Y gozlenen 819 gun %85'lik dilim (medyan 552, p95 1056, p99 1317). "
        "PLANLAMA RAKAMI: %95 kotu senaryo A2 icin maxDD %34 + 857 gun (2.3 yil) su alti, Y icin %35 + "
        "1056 gun (2.9 yil). Bu, canli kullanimda stratejiyi birakma riskinin asil kaynagi.",
    "S4_deflated_sharpe":
        "HESAP: defterde tsmom+tsmom2 ailelerinde TRAIN raporu olan BENZERSIZ (aile, ad, params) uclusu "
        "188; bunun 21'i kiyas/al-tut -> N_test=167 strateji konfigurasyonu. Bu 167 denemenin yillik train "
        "Sharpe'lari: ort 1.269, std 0.261, min 0.35, maks 1.60. "
        "Bailey & Lopez de Prado (2014): tum hesap GUNLUK frekansta. "
        "SR0 = sigma_SR * [(1-gamma)*z(1-1/N) + gamma*z(1-1/(N*e))], gamma=0.5772 (Euler), "
        "sigma_SR = 0.261/sqrt(365) = 0.01366 gunluk -> SR0 = 0.0370 gunluk = 0.706 YILLIK. "
        "Yani 167 tamamen degersiz deneme arasinda EN IYISININ, sirf sansla, yillik 0.71 Sharpe uretmesi "
        "beklenir - gozlenen 1.59 bunun iki kati. "
        "DSR = Phi[ (SR - SR0)*sqrt(n-1) / sqrt(1 - carpiklik*SR + (basiklik-1)/4*SR^2) ]. "
        "A2 train: n=2467 gun, SR=1.592 yillik (0.0833 gunluk), carpiklik +0.596, basiklik 13.40 "
        "-> z=2.33, DSR=0.9902. Yani 'sans' aciklamasinin olasiligi ~%1. Y train: SR=1.192, carpiklik +1.31, "
        "basiklik 22.5 -> DSR=0.9038 (~%10 sans olasiligi). gelistirme (n=2830) ile A2 0.9925 / Y 0.9348. "
        "N duyarliligi: N=50 -> SR0 0.594, DSR A2 0.996 / Y 0.946; N=400 -> SR0 0.779, DSR A2 0.984 / Y 0.866. "
        "DSR>=0.95 icin gerekli yillik Sharpe (N=167, train) = 1.33. "
        "VARSAYIMLAR VE ZAYIFLIKLARI (onemli): (1) 167 denemenin BAGIMSIZ oldugu varsayilir - degiller "
        "(ayni 3 coin, ust uste binen MA'lar, ayni 2830 gun); etkin N muhtemelen 20-40, bu DSR'yi yukari "
        "ceker; ama ayni korelasyon sigma_SR'yi de kucultur ve bu DSR'yi asiri iyimser yapar - iki etki "
        "zit yonlu, net isaret bilinmiyor. (2) DSR yalniz COKLU TEST'i duzeltir; SEPET SECIM yanliligini "
        "(tsmom2: BNB'yi geriye donuk secmek 0.2-0.6 Sharpe) ve data1d_all'un HAYATTA-KALMA yanliligini "
        "duzeltmez. A2'ye 0.4 Sharpe haircut uygulanirsa (durust tahmin 1.19) DSR 0.99 -> ~0.90'a duser, "
        "yani Y ile ayni yere gelir. (3) Defter yalniz BU repodaki testleri sayar; onceki kesifleri (kirilim, "
        "kesitsel, rejim, ict = 640 kayit daha) ve insan sezgisiyle elenen fikirleri saymaz -> gercek N daha "
        "buyuk. (4) IID gunluk getiri varsayimi; otokorelasyon icin duzeltme yapilmadi (blok bootstrap bu "
        "acigi ayri kapatiyor). SONUC: A2'nin train Sharpe'i SADECE coklu testle aciklanamaz (p~%1), ama "
        "sepet secimi + coklu test BIRLIKTE ele alinirsa sinirdadir (p~%10).",
    "S5_alt_donemler":
        "A2 UC DONEMDE DE ONDE, Y BIRINDE GERIDE. (Sharpe / maxDD, strateji vt30 - MA50 duz - al-tut) "
        "A2 2018-2020: 1.73/%21.7 - 1.49/%48.5 - 1.01/%77.2 | 2021-2022: 0.99/%27.4 - 1.47/%49.8 - "
        "0.89/%73.6 | 2023-2025: 1.78/%15.3 - 1.54/%28.9 - 1.19/%40.4. Sharpe'ta 3/3, maxDD'de 3/3 ustun. "
        "Y 2018-2020: 1.50/%20.4 - 1.16/%47.8 - 0.67/%83.3 | 2021-2022: 0.25/%27.1 - 0.42/%50.1 - "
        "0.31/%82.2 | 2023-2025: 1.56/%16.3 - 1.23/%30.4 - 0.88/%52.3. Sharpe'ta 2/3 (2021-2022'de "
        "al-tutun ALTINDA), maxDD'de 3/3. "
        "ONEMLI AYRINTI: en zor donem 2021-2022 ve orada vol hedefleme ZARARLI - A2 icin duz MA50 1.47 vs "
        "vt30 0.99, Y icin 0.42 vs 0.25. vt30'un ustunlugu 2018-2020 ve 2023-2025'ten geliyor. Yani 'vt30 > "
        "ma50' sonucu donem-dayanikli DEGIL; 'MA50 > al-tut' sonucu 3/3 donemde ve iki sepette de dayanikli. "
        "Bu, edge'in TREND FILTRESINDE oldugunu, vol hedeflemenin ise bir risk politikasi (ve kismen "
        "donem-sansi) oldugunu tsmom'un H2 bulgusuyla tutarli sekilde dogruluyor.",
    "S6_kaldirac_uygulanabilirlik":
        "SPOT UYGULANABILIR - CUNKU ZATEN KALDIRACSIZ. agirliklar2 brut pozisyonu 1.0'a normalize ettigi "
        "icin A2/Y hicbir gun %100'un ustune cikmiyor (maks gross 1.00, ortalama 0.27/0.24). "
        "cap 1.5 -> 1.0 (vol carpani tavani): A2 Sharpe 1.57->1.54, CAGR %34.9->%33.8, maxDD %27.5->%27.6; "
        "Y 1.23->1.21, CAGR %25.1->%24.1. Yani tavanin pratik etkisi yok. "
        "HEDEF VOL %20 (cap 1.5 veya 1.0 fark etmiyor, maks gross 0.82): A2 Sharpe 1.56, CAGR %22.6, "
        "maxDD %19.0, ciro/yil 13.9->9.3; Y Sharpe 1.23, CAGR %16.6, maxDD %18.9, ciro/yil 9.2. "
        "Vol hedefi saf bir OLCEK dugmesi: %30->%20 Sharpe'i degistirmiyor (1.57/1.56 ve 1.23/1.23), "
        "getiriyi ve dususu ~%35 kisiyor, ciroyu (dolayisiyla maliyet ve slippage riskini) %33 azaltiyor. "
        "KIYAS: vol hedeflemesiz duz MA50 A2 Sharpe 1.47 / CAGR %75.8 / maxDD %49.9, Y 0.93 / %37.6 / %50.2. "
        "SPOT ONERISI: hedef vol %20, cap 1.0 - kaldiracsiz, maks %82 yatirimli, beklenen maxDD ~%19 "
        "(MC %95 kotu senaryosu bu olcekte ~%24), CAGR ~%23 (A2) / ~%17 (Y).",
}

KARAR = (
    "KISMEN AYIRT EDILEBILIR - ama iddianin hangi yarisi oldugu onemli. "
    "(1) RISK yarisi (al-tuta gore daha dusuk maxDD): SANSTAN NET AYIRT EDILEBILIR. Blok bootstrap "
    "farki A2 -0.50, Y -0.56; GA99 sifiri disliyor, 2000 tekrarin hicbirinde fark pozitif degil. "
    "(2) GETIRI yarisi (daha yuksek Sharpe): SINIRDA. GA95 sifiri disliyor (A2 p=0.018, Y p=0.009), "
    "GA99 DISLAMIYOR. Deflated Sharpe (N=167 deneme, sigma_SR=0.261) A2 train 0.990 / Y 0.904 - yani "
    "coklu test tek basina A2'yi aciklayamiyor (~%1), Y icin ~%10. "
    "(3) AMA DSR sepet secim yanliligini duzeltmiyor; tsmom2'nin olctugu 0.2-0.6 Sharpe haircut'i "
    "uygulanirsa A2 da ~0.90'a duser. Sepeti gecmise bakmadan secen Y'nin 0.904'u bu yuzden DAHA DURUST "
    "bir sayidir ve o da klasik %95 esigini gecmiyor. "
    "(4) Parametre kirilganligi YOK (mukemmel yillik ongoru bile sadece +0.11..+0.26 Sharpe ekliyor, "
    "walk-forward sabit MA50'nin altinda kaliyor) - yani sonuc 'bir parametreye asiri uydurulmus' degil. "
    "Zayiflik parametre secimi degil, ORNEKLEM: 2830 gun / 8 yil ve ozunde 2 ayi + 2 boga rejimi = "
    "efektif 4-6 bagimsiz gozlem. "
    "OZET: 'MA50 trend filtresi al-tutun dususunu yariya indirir' sonucu saglam (%99). 'Bunu yaparken "
    "Sharpe'i da 0.5-0.6 arttirir' sonucu %95'te var, %99'da yok, ve coklu test + sepet secimi "
    "duzeltmelerinden sonra sinirda kaliyor. A2 ile Y'nin farki (1.57 vs 1.23) istatistiksel olarak "
    "anlamsiz; A2'nin ustunlugu buyuk olcude sepet secim yanliligidir. HOLDOUT 2026 bu soruyu tek basina "
    "cozmez (1 yil ~ 0.5 bagimsiz gozlem) ama IKISI BIRDEN kosulursa aralarindaki farkin devam edip "
    "etmedigi bilgi verir."
)

DUSENLER = [
    "walk-forward MA secimi (onceki 3 yilin en iyisi): 2021-2025 Sharpe A2 1.05 / Y 0.66 - sabit MA50'nin "
    "(1.48 / 1.08) belirgin ALTINDA. Adaptif parametre bu ailede zarar veriyor.",
    "yillik 'en iyi MA' fikri: seciler 20-200 arasi zipliyor (std 40-55 gun), ardisik yillar arasi "
    "tasinabilirlik yok.",
    "Y (yuruyen_top3) 2021-2022 alt doneminde al-tutun altinda (Sharpe 0.25 vs 0.31) - uc alt donemde 2/3.",
    "vol hedeflemenin (vt30) duz MA50'ye ustunlugu donem-dayanikli DEGIL: 2021-2022'de A2 0.99 vs 1.47, "
    "Y 0.25 vs 0.42 ile GERIDE.",
    "Sharpe farkinin GA99'u her iki adayda da sifiri iceriyor - %99 guvenle 'al-tuttan iyi getiri' "
    "denemiyor.",
]

OGRENILEN = [
    "Iki farkli iddia var ve istatistiksel gucleri cok farkli: DUSUS AZALTMA iddiasi (maxDD farki) 2000 "
    "bootstrap tekrarinin 2000'inde ayni yonde, GA99 sifiri disliyor; GETIRI iddiasi (Sharpe farki) sadece "
    "GA95'te ayakta. Bir trend filtresinin yaptigi is ozunde bir RISK donusumudur; 'alfa' kismi 8 yillik "
    "veride olculemeyecek kadar kucuk/gurultulu.",
    "Parametre asiri-uydurma primi olculdu: her yil o yilin en iyi MA'sini SECEBILSEYDIK 2018-2025 Sharpe "
    "1.57 -> 1.83 (+0.26, 8 secim). Bu, 'bir serbestlik derecesi ~+0.03 Sharpe sisirir' anlamina gelir ve "
    "defterdeki 167 denemenin urettigi SR0=0.71'lik sans esigiyle ayni buyukluk mertebesinde.",
    "Yilin en iyi MA'si zipliyor ama bu KIRILGANLIK degil DUZLUK isareti: MA20-MA200 bandinin tamami cogu "
    "yilda al-tutu geciyor, tepe noktasi gurultuyle yer degistiriyor. Kirilganlik testi 'komsu parametre "
    "kotu mu' diye sorar - cevap hayir; 'tepe yil yila ayni mi' diye sorarsa cevap hayir ama bunun "
    "uygulamaya maliyeti yok (sabit MA50 zaten optimumun 0.11-0.21 altinda).",
    "Asil risk DERINLIK degil SURE: maxDD %27.5 (MC %95 kotu senaryo %34) yonetilebilir gorunuyor, ama "
    "gozlenen 730 gun su alti kalma suresi MC dagiliminin %89'unda ve %95 kotu senaryo 857-1056 gun "
    "(2.3-2.9 yil). Canli kullanimda stratejinin terk edilme olasiligi, kayip olasiligindan yuksek.",
    "Deflated Sharpe coklu testi duzeltir, SECIM YANLILIGINI duzeltmez. A2 DSR 0.99 gorunuyor cunku SR=1.59 "
    "kullaniliyor; oysa bu 1.59'un 0.2-0.6'si BNB'yi geriye donuk secmekten geliyor (tsmom2). Yanlilik "
    "duzeltilmis A2 ile hic yanliligi olmayan Y ayni yere (DSR ~0.90) dusuyor - iki adayin GERCEK beklentisi "
    "ayni; aralarindaki 0.34 Sharpe farki bir bulgu degil, bir olcum artefakti.",
    "Uygulanabilirlik engeli yok: strateji zaten kaldiracsiz (maks gross 1.00), hedef vol %20 + cap 1.0 ile "
    "maks %82 yatirimli, spot hesapta birebir kosulabilir; Sharpe degismiyor (1.56/1.23), ciro %33 azaliyor.",
]

ONERI = (
    "1) HOLDOUT 2026'ya A2 ve Y'yi BIRLIKTE goturun ve ONCEDEN su beklentiyi yazin: yanlilik duzeltmesi "
    "sonrasi durust tahmin ikisi icin de yillik Sharpe ~1.0-1.2, maxDD %20-35. A2'nin Y'den belirgin iyi "
    "cikmasini BEKLEMEYIN - bu iki adayin farki istatistiksel olarak yok. "
    "2) HOLDOUT olcutunu simdiden sabitleyin: tek yilin Sharpe'i ~+-0.9 standart hatayla gelir, bu yuzden "
    "'Sharpe > al-tut ve maxDD < al-tutun yarisi' ikili sartini kullanin, ciplak Sharpe siralamasini degil. "
    "3) UYGULAMA icin hedef vol %20 + cap 1.0 varyantini secin (spot, kaldiracsiz, maks %82 yatirimli, "
    "Sharpe ayni, ciro %33 dusuk, maxDD %19). Kaldiracli/vadeli gerekmiyor - bu, funding ve tasfiye "
    "riskini tamamen ortadan kaldirir. "
    "4) Parametre arastirmasini KAPATIN: bu ailede 167 konfigurasyon denendi, SR0=0.71 ve her yeni deneme "
    "esigi yukseltiyor. Yillik adaptif MA secimi acikca zararli (walk-forward 1.05 vs sabit 1.48). "
    "5) Kalan tek anlamli istatistiksel is ORNEKLEM BUYUTMEK: (a) ayni MA50 kuralini kripto disi varliklara "
    "(hisse endeksi, emtia, FX gunluk) uygulayip ayni yonu bulup bulmadigina bakmak - bu, 8 yil / 4-6 "
    "bagimsiz rejim kisitini asmanin tek durust yolu; (b) 2017-2018 oncesi BTC/ETH verisi ekleyerek pencereyi "
    "uzatmak. Kripto ici yeni varyant uretmek bilgi eklemez. "
    "6) Canli kullanim notu: %95 kotu senaryo 2.3-2.9 YIL su alti. Stratejiyi kullanacak kisiye bu sure "
    "onceden yazili verilmelidir; aksi halde terk etme, DD'nin kendisinden buyuk risk."
)

NOTLAR = [
    "Bu ailede HICBIR yeni parametre aranmadi ve hicbir secim yapilmadi. tsmom2.kos2 / tsmom2.sepet_cfg "
    "import edilerek A2 ve Y'nin AYNI kurali istatistiksel testlerden gecirildi. Yil-bazinda MA taramasi "
    "(20..200) bir SECIM degil, kararlilik OLCUMUdur; secilen kural her yerde sabit MA50'dir.",
    "Tum testler SADECE 'gelistirme' bolumunde (2018-04-01 .. 2025-12-31, 2830 gun). HOLDOUT'a dokunulmadi; "
    "LAB_HOLDOUT ayarlanmadi, ham .npy okunmadi. Maliyet harness varsayilani (7bp/birim ciro).",
    "Blok bootstrap ESLESMIS: her tekrarda ayni blok indeksleri hem stratejiye hem KENDI sepetinin al-tutuna "
    "uygulanir, boylece piyasa rejimi ortak tutulur ve fark istatistigi izole edilir. GA'lar yuzdelik "
    "(percentile) GA'lardir; bias duzeltmesi (BCa) yapilmadi.",
    "Monte Carlo TAKVIM AYI bloklariyla permutasyondur (93 blok, yerine koymadan): blok icindeki gun sirasi "
    "ve tum getiri kumesi korunur, sadece aylarin sirasi degisir. Bu, vol kumelenmesini kismen korur ama "
    "aylar-arasi otokorelasyonu ve trend surekliligini kirar - trend stratejisi icin bu KOTUMSER bir null'dur.",
    "Deflated Sharpe'ta N_test defterden sayildi (tsmom+tsmom2, TRAIN raporu olan benzersiz (aile,ad,params) "
    "= 188, kiyas/al-tut 21 cikarildi -> 167). sigma_SR bu 167 denemenin train Sharpe std'sidir (0.261, "
    "yillik). Denemeler bagimsiz DEGIL; DSR bu nedenle kesin bir olasilik degil, buyukluk mertebesi "
    "gostergesidir. Ayrica DSR sepet secim yanliligini ve survivorship'i duzeltmez.",
    "DETERMINIZM: tsmom2._AY onbellegi yuruyen sepetin aylik secimini ILK cagrilan konfigurasyonun MA "
    "isinma maskesiyle donduruyor (mevcut tsmom2 davranisi). Fazlar arasi tutarlilik icin saglamlik.py "
    "her kosuda onbellegi once referans kuralla (MA50+vt30) isitir (_isit); bu yuzden MA taramasindaki tum "
    "MA'lar AYNI aylik evren uzerinde karsilastirilir. Bu secim olmadan Y'nin yillik Sharpe'lari ~0.02 "
    "oynuyor (ornek: 2020 icin MA30 2.98 vs MA40 2.96).",
    "Ay-blok Monte Carlo ve blok bootstrap ayni veriyi iki kez kullanir; birbirinin bagimsiz teyidi "
    "degildirler. Ikisinin ortak mesaji 'ornek kisa' - GA genislikleri (Sharpe farki icin ~1.3 birim) bunu "
    "dogrudan gosteriyor.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="hepsi",
                    choices=["hepsi", "bootstrap", "yil", "montecarlo", "dsr", "altdonem", "kaldirac", "rapor"])
    ap.add_argument("--mc", type=int, default=10000)
    a = ap.parse_args()
    if a.faz == "rapor":       # ham sonuclardan JSON'u yeniden uret (hesap tekrarlanmaz)
        with open(os.path.join(LAB, "_saglamlik_ham.json"), encoding="utf-8") as f:
            yaz(json.load(f))
        return
    P = T.panel()
    print(f"panel: {len(P['syms'])} sembol, {len(P['gun'])} gun, bas idx={P['bas']} "
          f"({datetime.fromtimestamp(P['gun'][P['bas']]/1e3, timezone.utc).date()}) | bolum={BOLUM}")
    _isit()
    res = {}
    if a.faz in ("hepsi", "bootstrap"):
        print("\n=== 1) BLOK BOOTSTRAP (30 gun, 2000) ===")
        res["bootstrap"] = faz_bootstrap()
    if a.faz in ("hepsi", "yil"):
        print("\n=== 2) YIL BAZINDA MA KARARLILIGI ===")
        res["yil"] = faz_yil()
    if a.faz in ("hepsi", "montecarlo"):
        print("\n=== 3) MONTE CARLO (ay bloklari) ===")
        res["montecarlo"] = faz_montecarlo(B=a.mc)
    if a.faz in ("hepsi", "dsr"):
        print("\n=== 4) DEFLATED SHARPE ===")
        res["dsr"] = faz_dsr()
    if a.faz in ("hepsi", "altdonem"):
        print("\n=== 5) ALT DONEMLER ===")
        res["altdonem"] = faz_altdonem()
    if a.faz in ("hepsi", "kaldirac"):
        print("\n=== 6) KALDIRAC / UYGULANABILIRLIK ===")
        res["kaldirac"] = faz_kaldirac()
    if a.faz == "hepsi":
        with open(os.path.join(LAB, "_saglamlik_ham.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        yaz(res)


if __name__ == "__main__":
    main()
