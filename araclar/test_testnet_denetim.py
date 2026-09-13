"""
testnet_trader.iptal_et ve testnet_denetim BIRIM TESTLERI (requests mock; gercek testnete ISTEK GITMEZ)
Calistirma: PYTHONIOENCODING=utf-8 python -m unittest araclar/test_testnet_denetim.py -v
"""
import json
import os
import sys
import tempfile
import time
import unittest
from urllib.parse import parse_qsl

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if KOK not in sys.path:
    sys.path.insert(0, KOK)

import testnet_trader as tt      # noqa: E402
import trend_testnet as tn       # noqa: E402
import testnet_denetim as td     # noqa: E402

EXINFO = {"symbols": [{"symbol": s, "quantityPrecision": 3, "pricePrecision": 2,
                       "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                                   {"filterType": "LOT_SIZE", "stepSize": "0.001", "maxQty": "1000"},
                                   {"filterType": "MARKET_LOT_SIZE", "maxQty": "1000"}]}
                      for s in ("BTCUSDT", "BNBUSDT")]}
SIMDI = int(time.time() * 1000)
ESKI = SIMDI - 48 * 3600 * 1000
YENI = SIMDI - 1 * 3600 * 1000


class Y:
    def __init__(self, v, k=200):
        self._v, self.status_code = v, k

    def json(self):
        return self._v


class API:
    def __init__(self, pozlar=None, emirler=None, algo=None, poz_hata=False):
        self.pozlar = pozlar or {}          # sembol -> (miktar, updateTime)
        self.emirler = emirler or []
        self.algo = algo or []
        self.poz_hata = poz_hata
        self.cagri = []

    def get(self, url, params=None, timeout=None):
        if "exchangeInfo" in url:
            return Y(EXINFO)
        if "ticker/price" in url:
            return Y({"price": "100"})
        return Y({}, 404)

    def request(self, method, url, headers=None, timeout=None):
        yol, _, q = url.partition("?")
        yol = yol.replace(tt.BASE, "")
        p = dict(parse_qsl(q))
        self.cagri.append((method, yol, p))
        if yol == "/fapi/v2/positionRisk":
            if self.poz_hata:
                return Y({"code": -1000}, 500)
            sec = [s for s in self.pozlar if "symbol" not in p or p["symbol"] == s]
            return Y([{"symbol": s, "positionAmt": str(self.pozlar[s][0]), "markPrice": "100",
                       "updateTime": self.pozlar[s][1]} for s in sec])
        if yol == "/fapi/v1/openOrders":
            return Y(self.emirler)
        if yol == "/fapi/v1/openAlgoOrders":
            return Y(self.algo)
        if method == "DELETE":
            return Y({"code": 200, "msg": "ok"})
        if yol == "/fapi/v1/order":
            return Y({"orderId": 1})
        return Y({}, 400)

    def emir(self):
        return [p for m, y, p in self.cagri if y == "/fapi/v1/order"]

    def silme(self):
        return [(y, p) for m, y, p in self.cagri if m == "DELETE"]


class Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._eski = (tt.requests, tn.requests, tt.KEY, tt.SECRET, tn.KEY, tn.SECRET,
                      tt.STATE, tt.TREND_STATE, tn.POZ_DOSYA)
        tt.KEY = tt.SECRET = tn.KEY = tn.SECRET = "x"
        tt.STATE = os.path.join(self.tmp, "testnet_orders.json")
        tt.TREND_STATE = os.path.join(self.tmp, "trend_state.json")
        tn.POZ_DOSYA = os.path.join(self.tmp, "positions.json")

    def tearDown(self):
        (tt.requests, tn.requests, tt.KEY, tt.SECRET, tn.KEY, tn.SECRET,
         tt.STATE, tt.TREND_STATE, tn.POZ_DOSYA) = self._eski

    def api(self, **kw):
        a = API(**kw)
        tt.requests = tn.requests = a
        return a

    def yaz(self, yol, veri):
        with open(yol, "w") as f:
            json.dump(veri, f)

    def trend(self, defter):
        self.yaz(tt.TREND_STATE, {"portfoy": {"testnet_defter": defter}})

    def ict_kaydi(self, sembol, miktar, yon="LONG"):
        self.yaz(tt.STATE, [{"zaman": "2026-08-30T14:15:53", "sembol": sembol, "dilim": "1h",
                             "strateji": "ict_long", "yon": yon, "basarili": True,
                             "sonuc": {"orderId": 5, "miktar": miktar}}])

    def kayitlar(self):
        if not os.path.exists(tt.STATE):
            return []
        with open(tt.STATE) as f:
            return json.load(f)

    # ---------------- iptal_et ----------------
    def test_iptal_api_hatasinda_hicbir_seye_dokunmaz(self):
        a = self.api(poz_hata=True)
        tt.iptal_et("BTCUSDT", "stopped")
        self.assertEqual(a.silme(), [])
        self.assertEqual(a.emir(), [])
        k = self.kayitlar()[-1]
        self.assertFalse(k["basarili"])
        self.assertIn("okunamadi", k["sonuc"]["hata"])

    def test_iptal_pozisyon_yoksa_sadece_emir_iptali(self):
        a = self.api(pozlar={})
        tt.iptal_et("BTCUSDT", "cancelled")
        self.assertEqual(len(a.silme()), 2)            # normal + algo
        self.assertEqual(a.emir(), [])

    def test_iptal_bnb_olayi_ict_payini_kapatir_trende_dokunmaz(self):
        """30 Agustos BNB: ICT girisi 12.02 LONG, hesapta 4.39 (4.19 ICT kalintisi + 0.2 trend)."""
        a = self.api(pozlar={"BNBUSDT": (4.39, ESKI)})
        self.trend({"BNBUSDT": 0.2})
        self.ict_kaydi("BNBUSDT", 12.02)
        tt.iptal_et("BNBUSDT", "stopped")
        e = a.emir()
        self.assertEqual(len(e), 1, e)
        self.assertEqual((e[0]["side"], e[0]["reduceOnly"]), ("SELL", "true"))
        self.assertAlmostEqual(float(e[0]["quantity"]), 4.19, places=6)

    def test_iptal_ict_miktarindan_fazlasini_kapatmaz(self):
        """Trend defteri eskimis (0 goruyor) ama ICT girisi 3.0 -> en fazla 3.0 kapatilir."""
        a = self.api(pozlar={"BTCUSDT": (5.0, ESKI)})
        self.trend({})
        self.ict_kaydi("BTCUSDT", 3.0)
        tt.iptal_et("BTCUSDT", "target_done")
        self.assertAlmostEqual(float(a.emir()[0]["quantity"]), 3.0, places=6)

    def test_iptal_short_ict_kalintisi(self):
        a = self.api(pozlar={"BTCUSDT": (-2.5, ESKI)})
        self.trend({})
        self.ict_kaydi("BTCUSDT", 2.5, yon="SHORT")
        tt.iptal_et("BTCUSDT", "stopped")
        e = a.emir()[0]
        self.assertEqual(e["side"], "BUY")
        self.assertAlmostEqual(float(e["quantity"]), 2.5, places=6)

    def test_iptal_sadece_trend_payi_varsa_kapatmaz(self):
        a = self.api(pozlar={"BTCUSDT": (0.2, ESKI)})
        self.trend({"BTCUSDT": 0.2})
        self.ict_kaydi("BTCUSDT", 1.0)
        tt.iptal_et("BTCUSDT", "cancelled")
        self.assertEqual(a.emir(), [])

    # ---------------- denetim ----------------
    def test_denetim_eski_sahipsiz_pozisyonu_kapatir(self):
        a = self.api(pozlar={"BNBUSDT": (4.39, ESKI)})
        self.trend({"BNBUSDT": 0.2})
        oz = td.denetle(kapat=True)
        e = a.emir()
        self.assertEqual(len(e), 1, oz)
        self.assertAlmostEqual(float(e[0]["quantity"]), 4.19, places=6)
        self.assertEqual(oz["islem"], 1)
        self.assertEqual(self.kayitlar()[-1]["tur"], "DENETIM")

    def test_denetim_rapor_modunda_yazmaz(self):
        a = self.api(pozlar={"BNBUSDT": (4.39, ESKI)})
        self.trend({"BNBUSDT": 0.2})
        oz = td.denetle(kapat=False)
        self.assertEqual(a.emir(), [])
        self.assertEqual(a.silme(), [])
        self.assertTrue(any(x.startswith("SAHIPSIZ") for x in oz["satirlar"]))

    def test_denetim_ict_aktif_pozisyona_dokunmaz(self):
        a = self.api(pozlar={"BTCUSDT": (3.0, ESKI)},
                     algo=[{"symbol": "BTCUSDT", "orderType": "STOP_MARKET", "createTime": ESKI}])
        self.yaz(tn.POZ_DOSYA, [{"symbol": "BTCUSDT", "status": "open", "sinyal_gonderildi": True}])
        oz = td.denetle(kapat=True)
        self.assertEqual(a.emir(), [])
        self.assertEqual(a.silme(), [])
        self.assertEqual(oz["uyari"], 0)

    def test_denetim_stopsuz_aktif_ict_uyarir(self):
        self.api(pozlar={"BTCUSDT": (3.0, ESKI)})
        self.yaz(tn.POZ_DOSYA, [{"symbol": "BTCUSDT", "status": "open", "sinyal_gonderildi": True}])
        oz = td.denetle(kapat=True)
        self.assertEqual(oz["uyari"], 1)

    def test_denetim_yeni_pozisyona_dokunmaz(self):
        a = self.api(pozlar={"BTCUSDT": (3.0, YENI)})
        oz = td.denetle(kapat=True)
        self.assertEqual(a.emir(), [])
        self.assertEqual(oz["islem"], 0)

    def test_denetim_bayat_emri_iptal_eder_yeniye_dokunmaz(self):
        a = self.api(pozlar={}, emirler=[
            {"symbol": "BTCUSDT", "type": "LIMIT", "time": ESKI, "updateTime": ESKI},
            {"symbol": "BNBUSDT", "type": "LIMIT", "time": YENI, "updateTime": YENI}])
        oz = td.denetle(kapat=True)
        sil = a.silme()
        self.assertTrue(sil and all(p.get("symbol") == "BTCUSDT" for _, p in sil), sil)
        self.assertEqual(oz["islem"], 1)

    def test_denetim_trend_payi_tamam(self):
        a = self.api(pozlar={"BTCUSDT": (0.2, ESKI)})
        self.trend({"BTCUSDT": 0.2})
        oz = td.denetle(kapat=True)
        self.assertEqual(a.emir(), [])
        self.assertEqual((oz["islem"], oz["uyari"]), (0, 0))

    def test_denetim_pozisyon_okunamazsa_dokunmaz(self):
        a = self.api(poz_hata=True, emirler=[{"symbol": "BTCUSDT", "time": ESKI}])
        oz = td.denetle(kapat=True)
        self.assertEqual(a.silme(), [])
        self.assertEqual(oz["hata"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
