"""
arkaplan.py BIRIM TESTLERI (requests TAMAMEN mock; ag istegi YOK)
=================================================================
 (i)    ema_geri / kirilim sinyal tespiti lab implementasyonuyla (araclar/lab/ema_geri.py, kirilim_gunluk.py) ESDEGER;
        artimli islem motoru harness.sim_islem ile, portfoy adimi harness.sim_portfoy (+tersine funding) ile ayni sonuc
 (ii)   ayni gun iki kez kosunca state dosyasi bayt bayt degismez; kacan gunler sirayla islenir; istek < 250
 (iii)  stop / TP / zaman asimi / kanal cikisi / gap iptali R hesaplari
 (iv)   haftalik yeniden dengeleme sadece pazartesi
 (v)    tersine short funding dusumu
 (vi)   funding carry esik histerezisi
 (vii)  bir stratejide istisna olunca digerleri devam eder
 (viii) Telegram metninde kacirilmamis < > yok
 (ix)   haftalik rapor cift gonderilmez

Calistirma: PYTHONIOENCODING=utf-8 python -m unittest araclar/test_arkaplan.py
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(KOK, "araclar", "lab")
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import arkaplan as ark  # noqa: E402

GUN = ark.GUN_MS


def sentetik(n, seed, t0="2024-01-01", drift=0.002, vol=0.035, qv_taban=(3e6, 3e7)):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n) + 0.01 * np.sin(np.arange(n) / 25.0)
    c = 10 * np.exp(np.cumsum(ret))
    o = np.concatenate([[c[0]], c[:-1]]) * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.02, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.02, n)))
    qv = rng.uniform(*qv_taban, n)
    ot = ark.gun_ms(t0) + np.arange(n, dtype=np.int64) * GUN
    return ot, o, h, l, c, qv


def lab_yukle():
    try:
        if LAB not in sys.path:
            sys.path.insert(0, LAB)
        import harness as H           # noqa
        import ema_geri as LE         # noqa
        import kirilim_gunluk as LK   # noqa
        return H, LE, LK
    except Exception as e:            # lab klasoru yoksa esdegerlik testleri atlanir
        raise unittest.SkipTest(f"lab modulleri yuklenemedi: {e}")


def motor_kos(se, cfg, ad, son_kesme=2):
    """tek coin, tum gunler gun gun; lab trig[n-2:]=False ile ayni olsun diye son 2 gunde yeni sinyal yok"""
    st = ark.yeni_strateji(ad)
    V = ark.Veri({"XUSDT": se}, ["XUSDT"], se.gun)
    n = len(se.gun)
    with mock.patch.object(ark, "KAPANAN_MAKS", 10 ** 6):
        for k, D in enumerate(se.gun):
            V.evren = ["XUSDT"] if k < n - son_kesme else []
            ark.islem_gun(st, V, D, cfg)
    return st


# ===================================================================== (i) lab esdegerligi
class T1LabEsdegerlik(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H, cls.LE, cls.LK = lab_yukle()

    def _ema_lab(self, h, l, c, qv):
        LE = self.LE
        v30 = LE.roll_mean(qv, 30)
        D = dict(c=c, h=h, l=l, atr=LE.atr14(h, l, c), liq=np.isfinite(v30) & (v30 >= LE.MIN_VOL), ema={})
        cfg = dict(tf="1d", ema=[20, 50], Y=0.15, P=7, R=20, cikis="tp3", filtre="f0")
        sig, _, _ = LE.strateji_sinyal(D, cfg, np.ones(len(c), bool))
        return sig

    def _kir_lab(self, ot, o, h, l, c, qv):
        a = np.column_stack([ot, o, h, l, c, qv]).astype(float)
        cfg = dict(N=40, M=20, k=3.0, yon="long", tp=5, cikis="kanal")
        return a, self.LK.sinyaller(self.LK.ozellikler(a), cfg)

    def test_ema_geri_sinyal_esdeger(self):
        toplam = 0
        for seed in (1, 2, 3):
            ot, o, h, l, c, qv = sentetik(700, seed)
            n = len(c)
            lab = [(s["i"], s["stop"], s["tp"]) for s in self._ema_lab(h, l, c, qv)]
            self.assertTrue(all(s_max == 60 for s_max in [x["max_days"] for x in self._ema_lab(h, l, c, qv)]))
            tam = [(x["i"], x["stop"], x["tp"]) for x in ark.ema_geri_sinyaller(h, l, c, qv) if x["i"] < n - 2]
            self.assertEqual([x[0] for x in lab], [x[0] for x in tam])
            np.testing.assert_allclose([x[1:] for x in lab], [x[1:] for x in tam], rtol=1e-12)
            # canli: her gun sadece o gune kadar olan dilimle (min 150 bar)
            se = ark.Seri.diziden(ot, o, h, l, c, qv)
            gun = []
            for j in range(n - 2):
                x = ark.EMA_CFG["sinyal"](se, j)
                if x:
                    gun.append((x["i"], x["stop"], x["tp"]))
            lab150 = [x for x in lab if x[0] + 1 >= ark.EG["min_bar"]]
            self.assertEqual([x[0] for x in lab150], [x[0] for x in gun])
            np.testing.assert_allclose([x[1:] for x in lab150], [x[1:] for x in gun], rtol=1e-9)
            toplam += len(lab150)
        self.assertGreater(toplam, 0, "sentetik seride hic sinyal yok; test anlamsiz")

    def test_kirilim_sinyal_ve_kanal_esdeger(self):
        toplam = 0
        for seed in (4, 5, 6):
            ot, o, h, l, c, qv = sentetik(700, seed)
            n = len(c)
            _, sig = self._kir_lab(ot, o, h, l, c, qv)
            se = ark.Seri.diziden(ot, o, h, l, c, qv)
            gun = {}
            for j in range(n - 2):
                x = ark.KIRILIM_CFG["sinyal"](se, j)
                if x:
                    gun[x["i"]] = x
            lab200 = [s for s in sig if s["i"] + 1 >= ark.KG["min_bar"]]
            self.assertEqual([s["i"] for s in lab200], sorted(gun))
            for s in lab200:
                self.assertAlmostEqual(s["stop"], gun[s["i"]]["stop"], places=9)
                self.assertAlmostEqual(s["tp"], gun[s["i"]]["tp"], places=9)
                beklenen = next((j for j in range(s["i"] + 2, n) if ark.kirilim_kanal_cikis(c, j)), None)
                self.assertEqual(s["cikis_i"], beklenen)
                self.assertEqual(s["max_days"], ark.KG["max_gun"])
            toplam += len(lab200)
        self.assertGreater(toplam, 0)

    def _islem_karsilastir(self, lab_tr, st, n_min=1):
        benim = [dict(t=ark.gun_ms(k["sinyal_gun"]) + GUN, res=k["res"], r=k["r"], net=k["net"]) for k in st["kapanan"]]
        self.assertEqual(len(lab_tr), len(benim))
        for a, b in zip(lab_tr, benim):
            self.assertEqual(a["t"], b["t"])
            self.assertEqual(a["res"], b["res"])
            self.assertAlmostEqual(a["r"], b["r"], places=5)
            self.assertAlmostEqual(a["net"], b["net"], places=5)
        return len(benim)

    def test_ema_geri_motor_sim_islem_ile_ayni(self):
        H = self.H
        toplam = 0
        for seed in (1, 2, 3):
            ot, o, h, l, c, qv = sentetik(700, seed)
            a = np.column_stack([ot, o, h, l, c, qv]).astype(float)
            sig = [s for s in self._ema_lab(h, l, c, qv) if s["i"] + 1 >= ark.EG["min_bar"]]
            lab_tr = H.sim_islem(a, sig, tf="1d")
            st = motor_kos(ark.Seri.diziden(ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d")
            toplam += self._islem_karsilastir(lab_tr, st)
        self.assertGreater(toplam, 3)

    def test_kirilim_motor_sim_islem_ile_ayni(self):
        H = self.H
        toplam = 0
        for seed in (4, 5, 6):
            ot, o, h, l, c, qv = sentetik(700, seed)
            a, sig = self._kir_lab(ot, o, h, l, c, qv)
            sig = [s for s in sig if s["i"] + 1 >= ark.KG["min_bar"]]
            lab_tr = H.sim_islem(a, sig, tf="1d")
            st = motor_kos(ark.Seri.diziden(ot, o, h, l, c, qv), ark.KIRILIM_CFG, "kirilim_1d")
            toplam += self._islem_karsilastir(lab_tr, st)
        self.assertGreater(toplam, 3)

    def test_portfoy_adimi_sim_portfoy_ve_tersine_funding_ile_ayni(self):
        H = self.H
        rng = np.random.default_rng(11)
        n, m = 40, 4
        O = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (n, m)), axis=0))
        O[17, 2] = np.nan                      # eksik gun (delist/kesinti)
        O[30:, 3] = np.nan                     # coin tamamen biter
        W = rng.normal(0, 0.2, (n, m))
        W[5:9] = 0.0
        eq, g = H.sim_portfoy(O, O, W, maliyet=0.0007, funding=0.0)
        S = np.array([sum(-v for k, v in enumerate(W[t]) if v < 0 and not np.isnan(O[t + 1, k])) for t in range(n - 2)])
        g_net = g - 0.0003 * S
        ot = ark.gun_ms("2026-01-01") + np.arange(n) * GUN
        seri = {}
        for k in range(m):
            ok = ~np.isnan(O[:, k])
            seri[f"S{k}"] = ark.Seri.diziden(ot[ok], O[ok, k], O[ok, k], O[ok, k], O[ok, k], np.ones(ok.sum()))
        gunler = [ark.gun_str(x) for x in ot]
        V = ark.Veri(seri, [], gunler)
        st = ark.yeni_strateji("tersine_haftalik")
        st["hedef"] = {gunler[t]: {f"S{k}": float(W[t, k]) for k in range(m) if W[t, k] != 0} for t in range(n)}
        benim = [ark.portfoy_adim(st, gunler[t + 2], ark.spot_getiri(V), 0.0007, 0.0003) for t in range(n - 2)]
        np.testing.assert_allclose(benim, g_net, rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(st["equity"], float(np.prod(1 + g_net)), places=10)


# ===================================================================== (iii) R hesaplari
def duz_seri(n=330, fiyat=100.0):
    ot = ark.gun_ms("2025-06-01") + np.arange(n, dtype=np.int64) * GUN
    o = np.full(n, fiyat); c = np.full(n, fiyat); h = np.full(n, fiyat + 1); l = np.full(n, fiyat - 1)
    return ot, o, h, l, c, np.full(n, 1e7)


class T3RHesabi(unittest.TestCase):
    def _kos(self, dizi, cfg, ad, son, stop=95.0, tp=115.0, i=250):
        se = ark.Seri.diziden(*dizi)
        st = ark.yeni_strateji(ad)
        st["pozisyonlar"] = [dict(sym="XUSDT", sinyal_gun=se.gun[i], durum="bekliyor", stop=stop, tp=tp, ref=100.0)]
        V = ark.Veri({"XUSDT": se}, [], se.gun)
        for D in se.gun[i + 1:son + 1]:
            ark.islem_gun(st, V, D, cfg)
        return st, se

    def test_stop(self):
        ot, o, h, l, c, qv = duz_seri()
        l[253] = 94.0
        st, se = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 256)
        k = st["kapanan"][0]
        e = 100.0 * (1 + ark.SLIP); risk = e - 95.0
        self.assertEqual((k["res"], k["cikis_gun"], k["giris_gun"]), ("stopped", se.gun[253], se.gun[251]))
        self.assertAlmostEqual(k["r"], -1.0)
        self.assertAlmostEqual(k["net"], -1.0 - ark.MALIYET_ISLEM / (risk / e), places=6)
        self.assertEqual(st["pozisyonlar"], [])
        self.assertAlmostEqual(st["toplam_net_r"], k["net"], places=6)

    def test_stop_giris_mumunda_da_gecerli_tp_gecmez(self):
        ot, o, h, l, c, qv = duz_seri()
        h[251] = 120.0                                   # giris mumunda TP sayilmaz
        st, _ = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 255)
        self.assertEqual(st["kapanan"], [])
        self.assertEqual(st["pozisyonlar"][0]["durum"], "acik")
        ot, o, h, l, c, qv = duz_seri()
        l[251] = 90.0                                    # giris mumunda stop sayilir (kotumser)
        st, _ = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 255)
        self.assertEqual(st["kapanan"][0]["res"], "stopped")

    def test_tp_ve_stop_ayni_mum_stop_once(self):
        ot, o, h, l, c, qv = duz_seri()
        h[252] = 116.0
        st, _ = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 255)
        e = 100.0 * (1 + ark.SLIP); risk = e - 95.0
        k = st["kapanan"][0]
        self.assertEqual(k["res"], "target_done")
        self.assertAlmostEqual(k["r"], (115.0 - e) / risk, places=6)
        self.assertAlmostEqual(k["net"], (115.0 - e) / risk - ark.MALIYET_ISLEM / (risk / e), places=6)
        ot, o, h, l, c, qv = duz_seri()
        h[252] = 116.0; l[252] = 94.0
        st, _ = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 255)
        self.assertEqual(st["kapanan"][0]["res"], "stopped")

    def test_zaman_asimi_60_gun(self):
        ot, o, h, l, c, qv = duz_seri()
        c[311] = 103.0; h[311] = 103.5                   # sinyal 250 -> j=312'de zaman asimi, c[311]'den
        st, se = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 311)
        self.assertEqual(st["kapanan"], [])              # 311. gun hala acik
        st, se = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 312)
        k = st["kapanan"][0]
        e = 100.0 * (1 + ark.SLIP); risk = e - 95.0
        self.assertEqual((k["res"], k["cikis_gun"]), ("timeout", se.gun[311]))
        self.assertAlmostEqual(k["r"], (103.0 - e) / risk, places=6)

    def test_kirilim_kanal_cikisi(self):
        ot, o, h, l, c, qv = duz_seri()
        c[251] = 98.0; l[251] = 97.5                     # giris gunu kanal cikisi sayilmaz
        c[255] = 97.0; l[255] = 96.5
        st, se = self._kos((ot, o, h, l, c, qv), ark.KIRILIM_CFG, "kirilim_1d", 258)
        k = st["kapanan"][0]
        e = 100.0 * (1 + ark.SLIP); risk = e - 95.0
        self.assertEqual((k["res"], k["cikis_gun"]), ("cikis", se.gun[255]))
        self.assertAlmostEqual(k["r"], (97.0 - e) / risk, places=6)

    def test_gap_iptal_kilitsiz(self):
        ot, o, h, l, c, qv = duz_seri()
        o[251] = 94.0
        st, _ = self._kos((ot, o, h, l, c, qv), ark.EMA_CFG, "ema_geri_1d", 253)
        self.assertEqual((st["iptal"], st["islem_sayisi"], st["pozisyonlar"]), (1, 0, []))


# ===================================================================== sahte borsa (ii, ix)
class Yanit:
    def __init__(self, veri, kod=200):
        self._v = veri
        self.status_code = kod
        self.headers = {}
        self.text = ""

    def json(self):
        return self._v


class SahteBorsa:
    def __init__(self, bitis="2026-09-30", n_gun=460, n_ek=23):
        self.simdi = None
        self.cagrilar = []
        self.postlar = []
        bit = ark.gun_ms(bitis)
        self.syms = list(ark.CEKIRDEK) + [f"K{k:02d}USDT" for k in range(n_ek)]
        self.spot, self.perp, self.fund = {}, {}, {}
        for r, s in enumerate(self.syms):
            ot, o, h, l, c, qv = sentetik(n_gun, 100 + r, t0=ark.gun_str(bit - (n_gun - 1) * GUN), drift=0.001)
            qv = qv / qv.mean() * (4e9 / (r + 1))
            self.spot[s] = [[int(ot[i]), f"{o[i]}", f"{h[i]}", f"{l[i]}", f"{c[i]}", "1000", int(ot[i]) + GUN - 1,
                             f"{qv[i]}", 1, "0", "0", "0"] for i in range(n_gun)]
            self.perp[s] = [[x[0], f"{float(x[1]) * 1.0004}", x[2], x[3], f"{float(x[4]) * 1.0004}", "500", x[6], x[7]]
                            for x in self.spot[s]]
            oran = 0.0004 if r % 3 == 0 else 0.00005    # yillik ~%44 vs ~%5.5
            self.fund[s] = [dict(fundingTime=int(ot[i]) + k * 8 * 3600_000 + 3, fundingRate=f"{oran}")
                            for i in range(n_gun) for k in (1, 2, 3)]

    def _su_an(self):
        return int(self.simdi.timestamp() * 1000)

    def get(self, url, params=None, timeout=None):
        p = dict(params or {})
        self.cagrilar.append((url, p))
        now = self._su_an()
        if url.endswith("/fapi/v1/exchangeInfo"):
            return Yanit({"symbols": [dict(symbol=s, quoteAsset="USDT", contractType="PERPETUAL", status="TRADING")
                                      for s in self.syms + ["PAXGUSDT"]]})
        if url.endswith("/api/v3/ticker/24hr"):
            return Yanit([dict(symbol=s, quoteVolume=str(1e9 / (k + 1))) for k, s in enumerate(self.syms)]
                         + [dict(symbol="PAXGUSDT", quoteVolume="9e12"), dict(symbol="USDCUSDT", quoteVolume="9e12")])
        if url.endswith("/klines"):
            kaynak = self.perp if "fapi" in url else self.spot
            if p.get("symbol") not in kaynak:
                return Yanit({"code": -1121}, 400)
            rows = [x for x in kaynak[p["symbol"]] if x[0] <= now]
            return Yanit(rows[-int(p.get("limit", 500)):])
        if url.endswith("/fapi/v1/fundingRate"):
            rows = [x for x in self.fund.get(p["symbol"], []) if int(p["startTime"]) <= x["fundingTime"] <= now]
            return Yanit(rows[:int(p.get("limit", 100))])
        raise AssertionError(f"beklenmeyen URL (ag istegi yasak): {url}")

    def post(self, url, data=None, timeout=None):
        self.postlar.append((url, dict(data or {})))
        return Yanit({"ok": True})


class _BorsaTest(unittest.TestCase):
    def setUp(self):
        self.borsa = SahteBorsa()
        self.tmp = tempfile.mkdtemp()
        self.state = os.path.join(self.tmp, "arkaplan_state.json")
        self.trend = os.path.join(self.tmp, "trend_state.json")
        self.ict = os.path.join(self.tmp, "positions.json")
        with open(self.trend, "w") as f:
            json.dump({"portfoy": {"b3": {"equity": 1.02, "altut": 1.05, "baslangic": "2026-09-11"},
                                   "yuruyen3": {"equity": 0.99, "altut": 0.97},
                                   "yuruyen5_tavan": {"equity": 1.0, "altut": 1.0}}}, f)
        yamalar = [mock.patch.object(ark.requests, "get", self.borsa.get),
                   mock.patch.object(ark.requests, "post", self.borsa.post),
                   mock.patch.object(ark, "_uyu", lambda s: None),
                   mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "1", "TOPIC_SUMMARY": "9"}),
                   mock.patch("builtins.print", lambda *a, **k: None)]
        for y in yamalar:
            y.start()
            self.addCleanup(y.stop)
        os.environ.pop("ARKAPLAN_HER_GUN", None)

    def kos(self, zaman, kuru=False):
        self.borsa.simdi = zaman
        return ark.kosu(kuru=kuru, simdi=zaman, state_yolu=self.state, trend_yolu=self.trend, ict_yolu=self.ict)

    def dosya(self):
        with open(self.state, "rb") as f:
            return f.read()


def utc(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


# ===================================================================== (ii) ayni gun iki kez
class T2CiftSaymaYok(_BorsaTest):
    def test_ayni_gun_iki_kosu_state_degismez(self):
        st, _, _ = self.kos(utc("2026-09-16 02:20"))            # carsamba; son kapanmis gun 09-15
        self.assertLess(len(self.borsa.cagrilar), 250)
        ilk = self.dosya()
        for ad in ark.STRATEJI_FONK:
            self.assertEqual(st["stratejiler"][ad]["son_gun"], "2026-09-15", ad)
            self.assertIsNone(st["stratejiler"][ad]["son_hata"], ad)
        self.kos(utc("2026-09-16 05:00"))
        self.assertEqual(ilk, self.dosya())

    def test_kacan_gunler_sirayla_ve_kuru_yazmaz(self):
        self.kos(utc("2026-09-16 02:20"))
        ilk = self.dosya()
        st, _, _ = self.kos(utc("2026-09-19 02:20"), kuru=True)
        self.assertEqual(ilk, self.dosya())                     # kuru: dosya yazilmadi
        self.assertEqual(self.borsa.postlar, [])
        self.assertEqual(st["stratejiler"]["funding_carry"]["son_gun"], "2026-09-18")
        st, _, _ = self.kos(utc("2026-09-19 02:20"))
        for ad in ark.STRATEJI_FONK:
            s = st["stratejiler"][ad]
            self.assertEqual(s["son_gun"], "2026-09-18", ad)
            self.assertEqual(s["baslangic"], "2026-09-15", ad)
            gunler = [x[0] for x in s["seri"]]
            self.assertEqual(gunler, ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"], ad)
        fc = st["stratejiler"]["funding_carry"]
        self.assertTrue(fc["acik"])                              # yuksek funding'li cekirdek coinler acik
        self.assertNotEqual(fc["equity"], 1.0)


# ===================================================================== (ix) haftalik rapor
class T9HaftalikRapor(_BorsaTest):
    def test_pazartesi_bir_kez(self):
        self.kos(utc("2026-09-14 02:20"))                       # pazartesi
        self.assertEqual(len(self.borsa.postlar), 1)
        url, d = self.borsa.postlar[0]
        self.assertEqual((d["parse_mode"], d["message_thread_id"]), ("HTML", "9"))
        self.assertIn("ARKA PLAN STRATEJILERI (bilgi amacli, kagit)", d["text"])
        self.assertIn("Bu stratejiler lab testlerinde elendi; ileriye donuk kanit icin izleniyor.", d["text"])
        self.kos(utc("2026-09-14 09:00"))                       # ayni pazartesi tekrar
        self.assertEqual(len(self.borsa.postlar), 1)
        self.kos(utc("2026-09-15 02:20"))                       # sali
        self.assertEqual(len(self.borsa.postlar), 1)
        self.kos(utc("2026-09-21 02:20"))                       # sonraki pazartesi
        self.assertEqual(len(self.borsa.postlar), 2)
        self.assertEqual(ark.state_oku(self.state)["son_rapor"], "2026-09-21")

    def test_her_gun_env(self):
        with mock.patch.dict(os.environ, {"ARKAPLAN_HER_GUN": "1"}):
            self.kos(utc("2026-09-15 02:20"))
            self.kos(utc("2026-09-15 03:20"))
        self.assertEqual(len(self.borsa.postlar), 2)


# ===================================================================== sentetik cok-coinli Veri (iv, vii)
def cok_coinli_veri(n_coin=36, n_gun=330, bitis="2026-09-20"):
    t0 = ark.gun_str(ark.gun_ms(bitis) - (n_gun - 1) * GUN)
    seri = {}
    syms = ["BTCUSDT"] + [f"K{k:02d}USDT" for k in range(n_coin - 1)]
    for r, s in enumerate(syms):
        ot, o, h, l, c, qv = sentetik(n_gun, 300 + r, t0=t0, drift=0.0015 if s == "BTCUSDT" else 0.0)
        seri[s] = ark.Seri.diziden(ot, o, h, l, c, qv / qv.mean() * 1e9 / (r + 1))
    return ark.Veri(seri, syms, seri["BTCUSDT"].gun, fund={}, perp={})


# ===================================================================== (iv) pazartesi
class T4Pazartesi(unittest.TestCase):
    def test_dengeleme_sadece_pazartesi(self):
        V = cok_coinli_veri()
        gunler = V.takvim[-16:]
        self.assertFalse(ark.pazartesi_mi(gunler[0]))
        for ad, fn in (("momentum_haftalik", ark.momentum_gun), ("tersine_haftalik", ark.tersine_gun)):
            st = ark.yeni_strateji(ad)
            onceki_baz = None
            dengeleme = []
            for k, D in enumerate(gunler):
                fn(st, V, D)
                if ark.pazartesi_mi(D):
                    dengeleme.append(D)
                    self.assertEqual(st["son_dengeleme"], D)
                    self.assertTrue(st["baz"], f"{ad}: pazartesi bos baz")
                elif onceki_baz is not None:
                    self.assertEqual(st["baz"], onceki_baz, f"{ad}: {D} pazartesi degil ama baz degisti")
                if k == 0:
                    self.assertEqual(st["hedef"][D], {}, "ilk pazartesiye kadar nakit")
                if ad == "momentum_haftalik":
                    beklenen = st["baz"] if ark.btc_rejim(V, D) == 1 else {}
                    self.assertEqual(st["hedef"][D], beklenen)
                else:
                    self.assertEqual(st["hedef"][D], st["baz"])
                self.assertLessEqual(len(st["hedef"]), 2)
                onceki_baz = dict(st["baz"])
            self.assertEqual(len(dengeleme), 2)
            self.assertTrue(all(ark.pazartesi_mi(g) for g in dengeleme))

    def test_agirlik_tanimlari(self):
        V = cok_coinli_veri()
        D = [g for g in V.takvim if ark.pazartesi_mi(g)][-1]
        bm = ark.momentum_baz(V, D)
        self.assertEqual(len(bm), 20)
        self.assertAlmostEqual(sum(bm.values()), 1.0)
        bt = ark.tersine_baz(V, D)
        self.assertEqual(sum(1 for v in bt.values() if v > 0), 10)
        self.assertEqual(sum(1 for v in bt.values() if v < 0), 10)
        self.assertAlmostEqual(sum(bt.values()), 0.0)
        self.assertAlmostEqual(sum(abs(v) for v in bt.values()), 1.0)
        # en kotu 3g getiri long, en iyi short
        x = {s: V.seri[s].c[V.seri[s].ix[D]] / V.seri[s].c[V.seri[s].ix[D] - 3] - 1 for s in bt}
        self.assertLess(max(x[s] for s in bt if bt[s] > 0), min(x[s] for s in bt if bt[s] < 0))


# ===================================================================== (v) tersine short funding
class T5ShortFunding(unittest.TestCase):
    def test_short_notional_funding_dusulur(self):
        st = ark.yeni_strateji("tersine_haftalik")
        W = {"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25}
        st["hedef"] = {"2026-09-01": W, "2026-09-02": W}
        sabit = lambda s, t: (True, 0.0)
        g1 = ark.portfoy_adim(st, "2026-09-03", sabit, 0.0007, short_funding=0.0003)
        self.assertAlmostEqual(g1, -1.0 * 0.0007 - 0.0003 * 0.5, places=12)
        x1 = -1.0 * 0.0007
        g2 = ark.portfoy_adim(st, "2026-09-04", sabit, 0.0007, short_funding=0.0003)
        surukleme = sum(abs(v - v / (1 + x1)) for v in W.values()) * 0.0007
        self.assertAlmostEqual(g2, -surukleme - 0.0003 * 0.5, places=12)
        self.assertAlmostEqual(st["equity"], (1 + g1) * (1 + g2), places=12)
        # long-only: funding yok
        st2 = ark.yeni_strateji("tersine_haftalik")
        st2["hedef"] = {"2026-09-01": {"A": 1.0}}
        self.assertAlmostEqual(ark.portfoy_adim(st2, "2026-09-03", sabit, 0.0007, 0.0003), -0.0007, places=12)


# ===================================================================== (vi) funding histerezis
class T6FundingHisterezis(unittest.TestCase):
    def test_saf_kural(self):
        yol = [(0.12, True), (0.07, True), (0.051, True), (0.049, False), (0.08, False), (0.10, False), (0.1001, True)]
        acik = False
        for f7, beklenen in yol:
            acik = ark.funding_esik(acik, f7)
            self.assertEqual(acik, beklenen, f"f7={f7}")

    def test_motor_ile(self):
        n = 60
        ot = ark.gun_ms("2026-07-01") + np.arange(n, dtype=np.int64) * GUN
        fiyat = np.full(n, 100.0)
        spot = ark.Seri.diziden(ot, fiyat, fiyat + 1, fiyat - 1, fiyat, np.full(n, 1e9))
        perp = ark.Seri.diziden(ot, fiyat, fiyat + 1, fiyat - 1, fiyat, np.full(n, 1e9))
        gunler = spot.gun
        fd = {}
        for k, g in enumerate(gunler):
            fd[g] = 0.0004 if k < 20 else (0.0002 if k < 35 else 0.0001)   # yillik %14.6 / %7.3 / %3.65
        V = ark.Veri({"BTCUSDT": spot}, [], gunler, fund={"BTCUSDT": fd}, perp={"BTCUSDT": perp})
        st = ark.yeni_strateji("funding_carry")
        durum = {}
        for g in gunler[10:50]:
            ark.funding_gun(st, V, g)
            durum[g] = bool(st["acik"].get("BTCUSDT"))
        self.assertTrue(durum[gunler[19]])
        self.assertTrue(durum[gunler[33]])        # %7.3: esik bandinda, acik kalir
        self.assertFalse(durum[gunler[45]])       # %3.65 < %5: kapanir
        self.assertEqual(st["hedef"][gunler[48]], {})
        # sifirdan baslangicta bant icindeki deger ACMAZ
        st2 = ark.yeni_strateji("funding_carry")
        ark.funding_gun(st2, V, gunler[33])
        self.assertEqual(st2["acik"], {})
        # acikken getiri = funding - maliyet (fiyat sabit): 0.5 notional * 0.0004
        st3 = ark.yeni_strateji("funding_carry")
        for g in gunler[10:14]:
            ark.funding_gun(st3, V, g)
        mal = ark.FND["maliyet"]
        g1 = 0.5 * 0.0004 - 0.5 * mal                        # giris maliyeti + 1 gun funding
        w_surukle = 0.5 * (1 + 0.0004) / (1 + g1)             # sim_portfoy agirlik suruklemesi
        g2 = 0.5 * 0.0004 - abs(0.5 - w_surukle) * mal        # hedefe geri dengeleme maliyeti
        self.assertAlmostEqual(st3["equity"], (1 + g1) * (1 + g2), places=12)


# ===================================================================== (vii) hata izolasyonu
class T7HataIzolasyonu(unittest.TestCase):
    def test_bir_strateji_patlarsa_digerleri_calisir(self):
        V = cok_coinli_veri()
        state = {}

        def patla(st, V, D):
            st["pozisyonlar"].append({"yarim": True})     # yarim kalan degisiklik geri alinmali
            raise ValueError("patladi <x>")

        with mock.patch.dict(ark.STRATEJI_FONK, {"kirilim_1d": patla}):
            ozet = ark.isle(state, V)
        tum = state["stratejiler"]
        self.assertIn("patladi", tum["kirilim_1d"]["son_hata"]["mesaj"])
        self.assertIsNone(tum["kirilim_1d"]["son_gun"])
        self.assertEqual(tum["kirilim_1d"]["pozisyonlar"], [])
        self.assertTrue(ozet["kirilim_1d"][0].startswith("HATA"))
        for ad in ("ema_geri_1d", "momentum_haftalik", "tersine_haftalik", "funding_carry"):
            self.assertEqual(tum[ad]["son_gun"], V.takvim[-1], ad)
            self.assertIsNone(tum[ad]["son_hata"], ad)
        # sonraki kosuda duzelirse normal devam eder ve hata temizlenir
        ark.isle(state, V)
        self.assertEqual(tum["kirilim_1d"]["son_gun"], V.takvim[-1])
        self.assertIsNone(state["stratejiler"]["kirilim_1d"]["son_hata"])

    def test_funding_verisi_yoksa_sadece_funding_hata(self):
        V = cok_coinli_veri()
        V.fund = None
        state = {}
        ark.isle(state, V)
        self.assertIsNotNone(state["stratejiler"]["funding_carry"]["son_hata"])
        self.assertIsNone(state["stratejiler"]["momentum_haftalik"]["son_hata"])


# ===================================================================== (viii) Telegram HTML
class T8TelegramKacis(unittest.TestCase):
    def test_kacirilmamis_isaret_yok(self):
        V = cok_coinli_veri()
        state = {"baslangic": V.takvim[-1], "baslangic_tarih": "2026-09-13"}
        ark.isle(state, V)
        state["stratejiler"]["tersine_haftalik"]["son_hata"] = {"gun": "x", "mesaj": "a<b> & <script>alert(1)</script>"}
        tmp = tempfile.mkdtemp()
        ict = os.path.join(tmp, "positions.json")
        with open(ict, "w") as f:
            json.dump([dict(symbol="A<B>USDT", status="stopped", opened_at="2026-09-14T01:00:00+00:00",
                            entry_bar_close="x", realized_r=-1.0, stop_dist_pct=0.02),
                       dict(symbol="C", status="target_done", opened_at="2026-09-12T01:00:00+00:00",
                            entry_bar_close="x", realized_r=5.0, stop_dist_pct=0.02),
                       dict(symbol="D", status="pending", opened_at="2026-09-14T02:00:00+00:00")], f)
        trend = os.path.join(tmp, "trend_state.json")
        with open(trend, "w") as f:
            json.dump({"portfoy": {"b3": {"equity": 1.1, "altut": 1.2, "baslangic": "<eski>"}}}, f)
        ark.okuyucu_guncelle(state, V, os.path.join(tmp, "yok.json"))   # trend_ref yok -> kendi baslangicindan
        metin = ark.rapor_metni(state, trend, ict)
        temiz = metin
        for et in ark.IZINLI_ETIKET:
            temiz = temiz.replace(et, "")
        self.assertNotIn("<", temiz)
        self.assertNotIn(">", temiz)
        self.assertIn("&lt;script&gt;", metin)
        self.assertIn("&lt;eski&gt;", metin)
        self.assertIn("-1.06R", metin)          # ICT: sadece baslangictan sonra acilan, net_r maliyetli
        self.assertIn("BTC al-tut ayni donem", metin)
        for ad in ark.STRATEJI_FONK:
            self.assertIn(ad, metin)


if __name__ == "__main__":
    unittest.main()
