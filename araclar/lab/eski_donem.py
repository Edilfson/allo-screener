"""eski_donem - BTC gunluk MA trend takibinin BINANCE ONCESI (2011-08 .. 2017-08) testi.

NEDEN: sonuc_saglamlik.json'un onerisi (madde 5b) -> "orneklemi buyut, 2017-2018 oncesi
BTC verisi ekleyerek pencereyi uzat". Bu donem hicbir parametre secimine girmedi: MA50,
hedef vol %30 / cap 1.5, sepet secimi - hepsi 2018-2025 verisiyle secildi. Dolayisiyla
2011-2017 GERCEK bir out-of-sample penceresidir.

VERI (public, anahtarsiz):
  Bitstamp  /api/v2/ohlc/btcusd  step=86400  -> 2011-08-18'den itibaren (ANA KAYNAK)
  Coinbase  exchange candles     granularity=86400 -> 2015-01'den itibaren (CAPRAZ KONTROL)
  (CryptoCompare artik 401 Unauthorized veriyor, Kraken sadece son ~720 gun -> kullanilmadi)
  .npy (ot_ms, o, h, l, c) olarak scratchpad/data_eski/ altina yazilir.

KURAL: gunluk kapanis > MA(N) -> long, degilse nakit. Islem t+1 acilista (harness.sim_portfoy
tam olarak bunu yapar: W[t] t+1 acilista uygulanir, getiri o[t+1]->o[t+2]).
  N in {20,30,50,75,100,150,200}
  maliyet tek yon %0.2 (o donemin borsalari pahali) ve %0.1
  varyantlar: temel (0/1) | hedef vol %20 ve %30, 30 gunluk gerceklesen vol, carpan tavani 1.0
  vol tabani 0.10 ve carpan tavani mantigi tsmom.py ile AYNI.

Kullanim:
  PYTHONIOENCODING=utf-8 python eski_donem.py --faz veri
  PYTHONIOENCODING=utf-8 python eski_donem.py --faz hepsi
"""
import argparse, json, math, os, sys, time, urllib.request
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

LAB = os.path.dirname(os.path.abspath(__file__))
SCRATCH = H.SCRATCH
DATA = os.path.join(SCRATCH, "data_eski")
AILE = "eski_donem"
BOLUM = "eski_2011_2017"          # harness bolum semasi disinda; sadece etiket
RNG = np.random.default_rng(20260913)

BAS = int(datetime(2011, 1, 1, tzinfo=timezone.utc).timestamp())
SON = int(datetime(2017, 9, 1, tzinfo=timezone.utc).timestamp())    # Binance BTCUSDT 2017-08-17
KES_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)   # data1d_all baslangici

MA_IZGARA = [20, 30, 50, 75, 100, 150, 200]
MALIYETLER = [0.002, 0.001]
VOL_TABAN = 0.10                   # tsmom.VOL_TABAN
ISINMA = 200                       # ortak analiz baslangici: en uzun MA isinsin (elmalar/elmalar)

AYI = [("ayi_2013_2014", "2013-12-01", "2014-12-31"),
       ("ayi_2014_2015", "2014-01-01", "2015-08-31")]


# =============================================================== 1) VERI
def _get(u, deneme=4):
    for k in range(deneme):
        try:
            r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(r, timeout=40).read())
        except Exception as e:
            if k == deneme - 1:
                raise
            time.sleep(2 + 2 * k)


def cek_bitstamp():
    """gunluk OHLC, 2011-08-18'den. Return {unix_sn: (o,h,l,c)}"""
    rows = {}
    start = BAS
    while True:
        d = _get(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={start}")["data"]["ohlc"]
        if not d:
            break
        for r in d:
            rows[int(r["timestamp"])] = (float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
        last = int(d[-1]["timestamp"])
        if last <= start or last > SON:
            break
        start = last + 86400
    return {t: v for t, v in rows.items() if t < SON}


def cek_coinbase():
    """gunluk OHLC 2015-01'den (300 mum/istek). Coinbase formati: [t, low, high, open, close, vol]"""
    rows = {}
    t0 = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp())
    while t0 < SON:
        t1 = min(t0 + 290 * 86400, SON)
        f = lambda x: datetime.fromtimestamp(x, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        d = _get("https://api.exchange.coinbase.com/products/BTC-USD/candles"
                 f"?granularity=86400&start={f(t0)}&end={f(t1)}")
        for r in d:
            rows[int(r[0])] = (float(r[3]), float(r[2]), float(r[1]), float(r[4]))
        t0 = t1
        time.sleep(0.4)
    return {t: v for t, v in rows.items() if t < SON}


def _npy(rows):
    ts = sorted(rows)
    return np.array([[t * 1000.0, *rows[t]] for t in ts], float)


def faz_veri(kaydet=True):
    os.makedirs(DATA, exist_ok=True)
    bs = cek_bitstamp()
    a_bs = _npy(bs)
    np.save(os.path.join(DATA, "BTCUSD_bitstamp.npy"), a_bs)
    try:
        cb = cek_coinbase()
        a_cb = _npy(cb)
        np.save(os.path.join(DATA, "BTCUSD_coinbase.npy"), a_cb)
    except Exception as e:
        cb, a_cb = {}, np.zeros((0, 5))
        print("coinbase FAIL", repr(e)[:150])

    d = lambda ms: datetime.fromtimestamp(ms / 1e3, timezone.utc).strftime("%Y-%m-%d")
    out = dict(
        bitstamp=dict(n=len(a_bs), bas=d(a_bs[0, 0]), son=d(a_bs[-1, 0])),
        coinbase=dict(n=len(a_cb), bas=d(a_cb[0, 0]) if len(a_cb) else None,
                      son=d(a_cb[-1, 0]) if len(a_cb) else None))

    # --- bosluk / bozuk bar kontrolu
    g = np.diff(a_bs[:, 0]) / 86400_000.0
    bosluk = [[d(a_bs[i, 0]), int(g[i])] for i in np.flatnonzero(g > 1)]
    c = a_bs[:, 4]
    sifir = int((c <= 0).sum())
    ext = np.abs(c[1:] / c[:-1] - 1)
    out["bitstamp"].update(eksik_gun=int(round(g.sum() - len(g))), bosluklar=bosluk[:20],
                           sifir_kapanis=sifir, en_buyuk_gunluk_hareket=round(float(ext.max()), 3),
                           gunluk_hareket_p999=round(float(np.quantile(ext, 0.999)), 3))

    # --- iki kaynak ortusen donemde tutarlilik
    if len(a_cb):
        ort = sorted(set(a_bs[:, 0].astype(np.int64)) & set(a_cb[:, 0].astype(np.int64)))
        mb = {int(x[0]): x[4] for x in a_bs}; mc = {int(x[0]): x[4] for x in a_cb}
        f = np.array([abs(mb[t] / mc[t] - 1) for t in ort])
        # sinyal seviyesinde: MA50 uzeri/alti kararlari kac gun ayrisiyor
        out["capraz_kontrol"] = dict(
            ortusen_gun=len(ort), bas=d(ort[0]), son=d(ort[-1]),
            kapanis_fark_ort_pct=round(float(f.mean() * 100), 4),
            kapanis_fark_medyan_pct=round(float(np.median(f) * 100), 4),
            kapanis_fark_p99_pct=round(float(np.quantile(f, 0.99) * 100), 4),
            kapanis_fark_maks_pct=round(float(f.max() * 100), 3),
            gun_fark_1pct_ustu=int((f > 0.01).sum()))
        # ayni kurali iki kaynakla kos -> sonuc ayni mi
        for src, arr in (("bitstamp", a_bs), ("coinbase", a_cb)):
            m = arr[:, 0] >= a_cb[0, 0]
            r = kos(arr[m], 50, 0.002)
            out["capraz_kontrol"][f"ma50_{src}_2015_2017"] = {k: r[k] for k in ("cagr", "maxdd", "sharpe", "islem")}
    if kaydet:
        H.kaydet(AILE, "veri_indirme", dict(kaynak="bitstamp+coinbase", bas="2011", son="2017-09"),
                 BOLUM, out, "veri kalitesi ve capraz kontrol")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:2500])
    return out


def yukle(kaynak="bitstamp", kes=True):
    a = np.load(os.path.join(DATA, f"BTCUSD_{kaynak}.npy"))
    if kes:
        a = a[a[:, 0] < KES_MS]        # Binance/data1d_all ile ORTUSME YOK -> tamamen bagimsiz
    return a


# =============================================================== 2) KURAL / SIMULASYON
def _roll_mean(x, N):
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out[N - 1:] = (cs[N:] - cs[:-N]) / N
    return out


def _roll_std(x, N):
    m1 = _roll_mean(x, N); m2 = _roll_mean(x * x, N)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0) * N / (N - 1))


def agirlik(a, N, hedef=None, cap=1.0, volwin=30, altut=False):
    """W[t]: t KAPANISINDA bilinen agirlik. Sinyal: close[t] > MA(N)[t]."""
    c = a[:, 4]
    ma = _roll_mean(c, N)
    r = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
    vol = np.full(len(c), np.nan)
    vol[1:] = _roll_std(np.nan_to_num(r[1:]), volwin) * math.sqrt(365)
    vol = np.maximum(vol, VOL_TABAN)
    w = np.ones(len(c)) if altut else (c > ma).astype(float)
    if hedef and not altut:
        w = w * np.clip(hedef / vol, 0, cap)
    w = np.minimum(w, 1.0)                      # gross <= 1 (harness/tsmom ile ayni)
    hazir = ~np.isnan(ma) & ~np.isnan(vol)
    w[~hazir] = 0.0
    return w


def kos(a, N, maliyet, hedef=None, cap=1.0, altut=False, bas=ISINMA, ad=None, kaydet=False, not_=""):
    """harness.sim_portfoy ile tek kolonlu O/C matrisi uzerinde kosar."""
    W = agirlik(a, N, hedef, cap, altut=altut)[bas:, None]
    O = a[bas:, 1][:, None]; C = a[bas:, 4][:, None]
    gun = a[bas:, 0]
    eq, g = H.sim_portfoy(O, C, W, maliyet=maliyet)
    gun = gun[:len(g)]
    r = ozet(g, gun)
    wv = W[:len(g), 0]
    prev = np.concatenate([[0.0], wv[:-1]])
    r["ciro_yil"] = round(float(np.abs(wv - prev).sum() / (len(g) / 365)), 2)
    r["islem"] = int(((wv > 1e-9) & (prev <= 1e-9)).sum())
    r["islem_yil"] = round(r["islem"] / (len(g) / 365), 1)
    r["ort_gross"] = round(float(wv.mean()), 3)
    r["piyasada_gun_pct"] = round(float((wv > 1e-9).mean() * 100), 1)
    r["_g"] = g; r["_gun"] = gun
    if kaydet:
        H.kaydet(AILE, ad or f"ma{N}_m{maliyet}", dict(ma=N, maliyet=maliyet, hedef=hedef, cap=cap,
                 altut=altut, kaynak="bitstamp", bas_index=bas), BOLUM,
                 {k: v for k, v in r.items() if not k.startswith("_")}, not_)
    return r


# =============================================================== olcumler
def sharpe(g):
    g = np.asarray(g, float)
    return float(g.mean() / g.std() * math.sqrt(365)) if g.std() > 0 and len(g) > 2 else 0.0


def maxdd(g):
    eq = np.cumprod(1 + np.asarray(g, float))
    pk = np.maximum.accumulate(eq)
    return float(((pk - eq) / pk).max())


def su_alti(g):
    eq = np.cumprod(1 + np.asarray(g, float)); pk = np.maximum.accumulate(eq)
    best = cur = 0
    for x in eq < pk:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


def ozet(g, gun=None):
    g = np.asarray(g, float); eq = np.cumprod(1 + g); yil = len(g) / 365
    d = lambda ms: datetime.fromtimestamp(ms / 1e3, timezone.utc).strftime("%Y-%m-%d")
    o = dict(gun=int(len(g)), toplam=round(float(eq[-1] - 1), 3),
             cagr=round(float(eq[-1] ** (1 / max(yil, 1e-9)) - 1), 4),
             maxdd=round(maxdd(g), 3), sharpe=round(sharpe(g), 2), su_alti_gun=su_alti(g))
    if gun is not None and len(gun):
        o["bas"] = d(gun[0]); o["son"] = d(gun[-1])
    return o


def yil_of(gun):
    return np.array([datetime.fromtimestamp(x / 1e3, timezone.utc).year for x in gun])


def yillik(g, gun):
    g = np.asarray(g); gun = np.asarray(gun)[:len(g)]; yl = yil_of(gun)
    out = {}
    for y in sorted(set(yl.tolist())):
        v = g[yl == y]
        eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
        out[int(y)] = dict(gun=int(len(v)), getiri=round(float(eq[-1] - 1), 3),
                           maxdd=round(float(((pk - eq) / pk).max()), 3), sharpe=round(sharpe(v), 2))
    return out


def _ms(s):
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def dilim(g, gun, a, b):
    gun = np.asarray(gun)[:len(g)]
    m = (gun >= _ms(a)) & (gun <= _ms(b))
    return np.asarray(g)[m], gun[m]


# =============================================================== 3) FAZLAR
def faz_tarama(kaydet=True):
    """Ana tablo: 7 MA x 2 maliyet x 3 varyant + al-tut. Tum donem + yil bazinda."""
    a = yukle()
    varyant = [("temel", dict()), ("vt20_cap10", dict(hedef=0.20)), ("vt30_cap10", dict(hedef=0.30))]
    out = {"maliyet": {}}
    for mal in MALIYETLER:
        bh = kos(a, 50, mal, altut=True, ad=f"altut_m{mal}", kaydet=kaydet, not_="al-tut kiyas")
        blok = dict(altut={k: v for k, v in bh.items() if not k.startswith("_")},
                    altut_yillar=yillik(bh["_g"], bh["_gun"]), konf={})
        for vad, kw in varyant:
            for N in MA_IZGARA:
                ad = f"ma{N}_{vad}_m{mal}"
                r = kos(a, N, mal, ad=ad, kaydet=kaydet, not_="2011-2017 tam donem", **kw)
                yl = yillik(r["_g"], r["_gun"])
                ylb = blok["altut_yillar"]
                gecti = sum(1 for y in yl if y in ylb and yl[y]["getiri"] > ylb[y]["getiri"])
                dd_iyi = sum(1 for y in yl if y in ylb and yl[y]["maxdd"] < ylb[y]["maxdd"])
                blok["konf"][ad] = dict({k: v for k, v in r.items() if not k.startswith("_")},
                                        yillar=yl, yil_sayisi=len(yl),
                                        altutu_gectigi_yil=gecti, dd_dusuk_yil=dd_iyi)
        out["maliyet"][str(mal)] = blok
        print(f"--- maliyet {mal:.3%} | al-tut CAGR {bh['cagr']:.3f} maxDD {bh['maxdd']:.3f} S={bh['sharpe']}")
        for ad, r in blok["konf"].items():
            print(f"  {ad:<24} CAGR {r['cagr']:>8.3f}  maxDD {r['maxdd']:>6.3f}  S={r['sharpe']:>5}  "
                  f"islem/yil {r['islem_yil']:>5}  gecti {r['altutu_gectigi_yil']}/{r['yil_sayisi']}  "
                  f"ddIyi {r['dd_dusuk_yil']}/{r['yil_sayisi']}")
    return out


def faz_ayi(kaydet=True):
    """2013-14 ve 2014-15 ayi piyasalarinda davranis + al-tutun en derin dususleri."""
    a = yukle()
    out = {}
    for mal in MALIYETLER:
        bh = kos(a, 50, mal, altut=True)
        for ad_, A, B in AYI:
            gb, gunb = dilim(bh["_g"], bh["_gun"], A, B)
            blok = dict(donem=[A, B], altut=ozet(gb, gunb))
            for vad, kw in (("temel", {}), ("vt30_cap10", dict(hedef=0.30))):
                for N in (20, 50, 100, 200):
                    r = kos(a, N, mal, **kw)
                    gs, guns = dilim(r["_g"], r["_gun"], A, B)
                    o = ozet(gs, guns)
                    w = agirlik(a, N, kw.get("hedef"), 1.0)[ISINMA:][:len(r["_g"])]
                    mm = (np.asarray(r["_gun"]) >= _ms(A)) & (np.asarray(r["_gun"]) <= _ms(B))
                    o["piyasada_gun_pct"] = round(float((w[mm] > 1e-9).mean() * 100), 1)
                    o["ort_gross"] = round(float(w[mm].mean()), 3)
                    blok[f"ma{N}_{vad}"] = o
                    if kaydet:
                        H.kaydet(AILE, f"{ad_}_ma{N}_{vad}_m{mal}",
                                 dict(ma=N, maliyet=mal, donem=[A, B], **kw), BOLUM, o, "ayi piyasasi dilimi")
            out[f"{ad_}_m{mal}"] = blok
            print(f"{ad_} m={mal}: altut {blok['altut']['toplam']:+.3f} (dd {blok['altut']['maxdd']:.3f}) | "
                  + " ".join(f"{k}={blok[k]['toplam']:+.3f}" for k in blok if k.startswith("ma")))
    # en derin al-tut dususleri icinde strateji ne yapti
    bh = kos(a, 50, 0.002, altut=True); r50 = kos(a, 50, 0.002, hedef=0.30)
    out["altut_en_derin_dususler"] = _dd_karsilastir(bh["_g"], r50["_g"], bh["_gun"], k=5)
    return out


def _dd_karsilastir(gb, gs, gun, k=5):
    eq = np.concatenate([[1.0], np.cumprod(1 + np.asarray(gb))])
    es = np.concatenate([[1.0], np.cumprod(1 + np.asarray(gs))])
    t = np.concatenate([[gun[0]], np.asarray(gun)[:len(gb)]])
    ep = []; pk = eq[0]; pki = 0; dip = eq[0]; dipi = 0; icinde = False
    for i in range(1, len(eq)):
        if eq[i] >= pk:
            if icinde:
                ep.append((pki, dipi, (pk - dip) / pk)); icinde = False
            pk = eq[i]; pki = i; dip = eq[i]; dipi = i
        else:
            if not icinde:
                icinde = True; dip = eq[i]; dipi = i
            if eq[i] < dip:
                dip = eq[i]; dipi = i
    if icinde:
        ep.append((pki, dipi, (pk - dip) / pk))
    ep.sort(key=lambda x: -x[2])
    f = lambda i: datetime.fromtimestamp(t[i] / 1e3, timezone.utc).strftime("%Y-%m-%d")
    return [dict(tepe=f(p), dip=f(d), altut_dusus=round(float(dep), 3),
                 strateji_ayni_pencerede=round(float(es[d] / es[p] - 1), 3), gun=int(d - p))
            for p, d, dep in ep[:k]]


def _mat_sharpe(X):
    s = X.std(1)
    return np.where(s > 0, X.mean(1) / s * math.sqrt(365), 0.0)


def _mat_maxdd(X):
    eq = np.cumprod(1 + X, axis=1); pk = np.maximum.accumulate(eq, axis=1)
    return ((pk - eq) / pk).max(1)


def blok_bootstrap(gs, gb, blok=30, B=2000, parca=250):
    """ESLESMIS hareketli blok bootstrap (saglamlik.py ile AYNI kod mantigi)."""
    n = len(gs); nb = int(math.ceil(n / blok)); ofs = np.arange(blok)
    d_sh = np.empty(B); d_dd = np.empty(B)
    for a in range(0, B, parca):
        b = min(a + parca, B)
        st = RNG.integers(0, n - blok + 1, size=(b - a, nb))
        idx = (st[:, :, None] + ofs[None, None, :]).reshape(b - a, nb * blok)[:, :n]
        Xs = gs[idx]; Xb = gb[idx]
        d_sh[a:b] = _mat_sharpe(Xs) - _mat_sharpe(Xb)
        d_dd[a:b] = _mat_maxdd(Xs) - _mat_maxdd(Xb)
    q = lambda x, p: round(float(np.quantile(x, p)), 3)
    return dict(B=B, blok_gun=blok, n_gun=int(n),
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
    a = yukle()
    out = {}
    for mal in MALIYETLER:
        bh = kos(a, 50, mal, altut=True)
        for vad, kw in (("temel", {}), ("vt30_cap10", dict(hedef=0.30)), ("vt20_cap10", dict(hedef=0.20))):
            for N in (20, 50, 100, 200):
                r = kos(a, N, mal, **kw)
                gs = np.asarray(r["_g"]); gb = np.asarray(bh["_g"])
                n = min(len(gs), len(gb))
                res = blok_bootstrap(gs[:n], gb[:n])
                ad = f"boot_ma{N}_{vad}_m{mal}"
                out[ad] = res
                if kaydet:
                    H.kaydet(AILE, ad, dict(ma=N, maliyet=mal, blok=30, B=2000, **kw), BOLUM, res,
                             "blok bootstrap: strateji - altut")
                print(f"{ad:<26} dS={res['gozlenen']['sharpe_fark']:>6} GA95={res['sharpe_fark']['ga95']} "
                      f"p={res['sharpe_fark']['p_fark_sifir_veya_alti']:<7} "
                      f"dDD={res['gozlenen']['maxdd_fark']:>6} GA95={res['maxdd_fark']['ga95']}")
    return out


def faz_komsu(kaydet=True):
    """MA50 komsularinin tutarliligi: 20..200 tam izgara, iki maliyet, uc varyant."""
    a = valid = yukle()
    out = {}
    grid = list(range(20, 201, 10))
    for mal in MALIYETLER:
        bh = kos(a, 50, mal, altut=True)
        for vad, kw in (("temel", {}), ("vt30_cap10", dict(hedef=0.30))):
            tab = {}
            for N in grid:
                r = kos(a, N, mal, ad=f"komsu_ma{N}_{vad}_m{mal}", kaydet=kaydet,
                        not_="MA izgarasi 20..200", **kw)
                tab[N] = dict(sharpe=r["sharpe"], cagr=r["cagr"], maxdd=r["maxdd"])
            sh = np.array([tab[N]["sharpe"] for N in grid])
            out[f"{vad}_m{mal}"] = dict(
                izgara=tab, altut_sharpe=bh["sharpe"], altut_maxdd=bh["maxdd"],
                sharpe_min=float(sh.min()), sharpe_maks=float(sh.max()),
                sharpe_ort=round(float(sh.mean()), 3), sharpe_std=round(float(sh.std()), 3),
                altutu_gecen_ma_sayisi=int((sh > bh["sharpe"]).sum()), toplam_ma=len(grid),
                en_iyi_ma=int(grid[int(sh.argmax())]),
                ma50_sirasi=int(1 + (sh > tab[50]["sharpe"]).sum()),
                ma50_ile_en_iyi_fark=round(float(sh.max() - tab[50]["sharpe"]), 3))
            print(f"{vad} m={mal}: altut S={bh['sharpe']} | MA izgara S {sh.min():.2f}..{sh.max():.2f} "
                  f"(ort {sh.mean():.2f}) | altutu gecen {int((sh > bh['sharpe']).sum())}/{len(grid)} | "
                  f"en iyi MA{grid[int(sh.argmax())]} | MA50 sirasi {1 + (sh > tab[50]['sharpe']).sum()}")
    return out


def faz_kiyas(kaydet=True):
    """AYNI kod, AYNI kural -> 2018-2025 BTC (data1d_all) uzerinde. Yan yana koymak icin.
    HOLDOUT'a dokunulmaz: harness.veri_1d zaten >=2026'yi fiziksel keser."""
    out = {}
    try:
        b = H.veri_1d("BTCUSDT")            # (ot,o,h,l,c,v), 2017-08-17'den, <2026 kesik
        a_new = b[:, :5]
    except Exception as e:
        return {"hata": repr(e)[:200]}
    a_old = yukle()
    for etiket, arr in (("eski_2011_2017", a_old), ("yeni_2018_2025", a_new)):
        blok = {}
        for mal in (0.002, 0.001, 0.0007):
            bh = kos(arr, 50, mal, altut=True)
            blok[f"altut_m{mal}"] = {k: v for k, v in bh.items() if not k.startswith("_")}
            for vad, kw in (("temel", {}), ("vt30_cap10", dict(hedef=0.30)), ("vt20_cap10", dict(hedef=0.20))):
                r = kos(arr, 50, mal, ad=f"kiyas_{etiket}_ma50_{vad}_m{mal}", kaydet=kaydet,
                        not_="donem karsilastirmasi", **kw)
                blok[f"ma50_{vad}_m{mal}"] = {k: v for k, v in r.items() if not k.startswith("_")}
                blok[f"ma50_{vad}_m{mal}"]["yillar"] = yillik(r["_g"], r["_gun"])
        # ayni blok bootstrap iki donemde de -> yan yana
        bh = kos(arr, 50, 0.002, altut=True)
        for vad, kw in (("temel", {}), ("vt30_cap10", dict(hedef=0.30))):
            r = kos(arr, 50, 0.002, **kw)
            n = min(len(r["_g"]), len(bh["_g"]))
            res = blok_bootstrap(np.asarray(r["_g"])[:n], np.asarray(bh["_g"])[:n])
            blok[f"bootstrap_ma50_{vad}_m0.002"] = res
            if kaydet:
                H.kaydet(AILE, f"kiyas_boot_{etiket}_ma50_{vad}", dict(ma=50, maliyet=0.002, **kw),
                         BOLUM, res, "donem karsilastirmasi bootstrap")
        out[etiket] = blok
    # sonuc_saglamlik.json'daki 3-coin A2 sonucu (referans)
    try:
        s = json.load(open(os.path.join(LAB, "sonuc_saglamlik.json"), encoding="utf-8"))
        e = s["en_iyi_3"][0]
        out["referans_saglamlik_A2"] = dict(ad=e["ad"], gelistirme=e["gelistirme"],
                                            altut=e["altut_gelistirme"], saglamlik=e["saglamlik"])
    except Exception as ex:
        out["referans_saglamlik_A2"] = repr(ex)[:120]
    return out


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faz", default="hepsi")
    ap.add_argument("--no-kaydet", action="store_true")
    A = ap.parse_args()
    k = not A.no_kaydet
    R = {}
    if A.faz in ("veri", "hepsi"):
        R["veri"] = faz_veri(k)
    if A.faz in ("tarama", "hepsi"):
        R["tarama"] = faz_tarama(k)
    if A.faz in ("ayi", "hepsi"):
        R["ayi"] = faz_ayi(k)
    if A.faz in ("komsu", "hepsi"):
        R["komsu"] = faz_komsu(k)
    if A.faz in ("bootstrap", "hepsi"):
        R["bootstrap"] = faz_bootstrap(k)
    if A.faz in ("kiyas", "hepsi"):
        R["kiyas"] = faz_kiyas(k)
    p = os.path.join(SCRATCH, f"_eski_donem_{A.faz}.json")
    json.dump(R, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("yazildi:", p)


if __name__ == "__main__":
    main()
