"""
trend_testnet.py BIRIM TESTLERI (requests mock'lanir, gercek testnete ISTEK GITMEZ)
===================================================================================
Senaryolar:
  1) sifirdan giris      : pozisyon 0, hedef %14 -> MARKET BUY, reduceOnly YOK
  2) kismi azaltma       : pozisyon 1.4, hedef %5 -> MARKET SELL reduceOnly=true
  3) degisiklik yok      : pozisyon == hedef -> emir gonderilmez (min notional atlama)
  4) ICT emri varsa atla : son 24 saatte basarili ICT giris kaydi olan sembol atlanir
  5) kuru mod            : hicbir imzali istek ve dosya yazimi olmaz

Calistirma: PYTHONIOENCODING=utf-8 python -m unittest araclar.test_trend_testnet -v
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

import testnet_trader as tt          # noqa: E402
import trend_testnet as tn           # noqa: E402

EXINFO = {"symbols": [{"symbol": "BTCUSDT", "quantityPrecision": 3, "pricePrecision": 2,
                       "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                                   {"filterType": "LOT_SIZE", "stepSize": "0.001",
                                    "maxQty": "1000"},
                                   {"filterType": "MARKET_LOT_SIZE", "maxQty": "120"}]}]}


class SahteYanit:
    def __init__(self, veri, kod=200):
        self._veri = veri
        self.status_code = kod

    def json(self):
        return self._veri


class SahteAPI:
    """requests yerine gecer: get = herkese acik uclar, request = imzali uclar."""

    def __init__(self, poz=0.0, son_fiyat=100.0):
        self.poz = poz
        self.son_fiyat = son_fiyat
        self.cagrilar = []           # (method, yol, params)

    # --- herkese acik ---
    def get(self, url, params=None, timeout=None):
        self.cagrilar.append(("GET", url.replace(tn.BASE, ""), dict(params or {})))
        if "/fapi/v1/exchangeInfo" in url:
            return SahteYanit(EXINFO)
        if "/fapi/v1/ticker/price" in url:
            return SahteYanit({"symbol": (params or {}).get("symbol"),
                               "price": str(self.son_fiyat)})
        return SahteYanit({"hata": "bilinmeyen uc"}, 404)

    # --- imzali ---
    def request(self, method, url, headers=None, timeout=None):
        yol, _, sorgu = url.partition("?")
        yol = yol.replace(tn.BASE, "")
        p = dict(parse_qsl(sorgu))
        self.cagrilar.append((method, yol, p))
        if yol == "/fapi/v2/positionRisk":
            return SahteYanit([{"symbol": p.get("symbol"), "positionAmt": str(self.poz),
                                "entryPrice": "100.0", "unRealizedProfit": "0",
                                "markPrice": str(self.son_fiyat)}])
        if yol == "/fapi/v1/leverage":
            return SahteYanit({"symbol": p.get("symbol"), "leverage": int(p.get("leverage", 0))})
        if yol == "/fapi/v1/order":
            return SahteYanit({"orderId": 999, "symbol": p.get("symbol"),
                               "status": "NEW", "type": p.get("type")})
        return SahteYanit({"hata": "bilinmeyen uc"}, 400)

    def emirler(self):
        return [p for m, yol, p in self.cagrilar if yol == "/fapi/v1/order"]


class TrendTestnetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.kayit_dosya = os.path.join(self.tmp, "testnet_orders.json")
        self._eski = (tn.requests, tt.requests, tt.KEY, tt.SECRET, tn.KEY, tn.SECRET,
                      tt.STATE, tn.TABAN, tn.POZ_DOSYA)
        tt.KEY = tn.KEY = "sahte_anahtar"
        tt.SECRET = tn.SECRET = "sahte_gizli"
        tt.STATE = self.kayit_dosya
        tn.TABAN = 1000.0
        tn.POZ_DOSYA = os.path.join(self.tmp, "positions.json")

    def tearDown(self):
        (tn.requests, tt.requests, tt.KEY, tt.SECRET, tn.KEY, tn.SECRET,
         tt.STATE, tn.TABAN, tn.POZ_DOSYA) = self._eski

    def _api(self, poz=0.0, son_fiyat=100.0):
        api = SahteAPI(poz=poz, son_fiyat=son_fiyat)
        tn.requests = api
        tt.requests = api
        return api

    def _kayitlar(self):
        if not os.path.exists(self.kayit_dosya):
            return []
        with open(self.kayit_dosya) as f:
            return json.load(f)

    # ---------------- 1) sifirdan giris ----------------
    def test_sifirdan_giris(self):
        api = self._api(poz=0.0)
        ozet = tn.esle({"BTCUSDT": 0.14})
        emirler = api.emirler()
        self.assertEqual(len(emirler), 1, emirler)
        e = emirler[0]
        self.assertEqual(e["side"], "BUY")
        self.assertEqual(e["type"], "MARKET")
        self.assertAlmostEqual(float(e["quantity"]), 1.4, places=6)   # 1000*0.14/100
        self.assertNotIn("reduceOnly", e)                              # giris emri
        self.assertEqual(ozet["emir"], 1)
        self.assertEqual(ozet["hata"], 0)
        # kaldirac 1x ayarlanmis olmali
        kaldirac = [p for m, yol, p in api.cagrilar if yol == "/fapi/v1/leverage"]
        self.assertEqual(kaldirac[0]["leverage"], "1")
        # kayit dosyasi
        k = self._kayitlar()[-1]
        self.assertEqual((k["tur"], k["sembol"], k["basarili"]), ("TREND", "BTCUSDT", True))
        self.assertAlmostEqual(k["hedef_notional"], 140.0, places=2)
        self.assertAlmostEqual(k["mevcut_miktar"], 0.0, places=6)
        self.assertAlmostEqual(k["delta"], 1.4, places=6)
        self.assertAlmostEqual(ozet["defter"]["BTCUSDT"], 1.4, places=6)   # defter guncellendi

    # ---------------- 2) kismi azaltma ----------------
    def test_kismi_azaltma_reduce_only(self):
        api = self._api(poz=1.4)
        ozet = tn.esle({"BTCUSDT": 0.05}, defter={"BTCUSDT": 1.4})   # hedef 50 USDT -> 0.5 adet
        emirler = api.emirler()
        self.assertEqual(len(emirler), 1, emirler)
        e = emirler[0]
        self.assertEqual(e["side"], "SELL")
        self.assertEqual(e["type"], "MARKET")
        self.assertAlmostEqual(float(e["quantity"]), 0.9, places=6)   # 1.4 -> 0.5
        self.assertEqual(e.get("reduceOnly"), "true")
        self.assertEqual(ozet["emir"], 1)
        k = self._kayitlar()[-1]
        self.assertAlmostEqual(k["delta"], -0.9, places=6)
        self.assertTrue(k["sonuc"]["reduceOnly"])

    # ---------------- 3) degisiklik yok ----------------
    def test_degisiklik_yok_atla(self):
        api = self._api(poz=1.4)
        ozet = tn.esle({"BTCUSDT": 0.14}, defter={"BTCUSDT": 1.4})   # hedef 1.4 = bizim 1.4
        self.assertEqual(api.emirler(), [])
        self.assertEqual(ozet["emir"], 0)
        self.assertEqual(ozet["atlanan"], 1)
        k = self._kayitlar()[-1]
        self.assertIsNone(k["basarili"])
        self.assertEqual(k["sonuc"]["atlandi"], "min notional")

    def test_kucuk_delta_atla(self):
        """0.1 adet = 10 USDT < 25 USDT -> emir gonderilmez."""
        api = self._api(poz=1.3)
        ozet = tn.esle({"BTCUSDT": 0.14}, defter={"BTCUSDT": 1.3})
        self.assertEqual(api.emirler(), [])
        self.assertEqual(ozet["atlanan"], 1)

    # ---------------- 4) ICT emri varsa atla ----------------
    def test_ict_emri_varsa_atla(self):
        api = self._api(poz=0.0)
        json.dump([{"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "sembol": "BTCUSDT",
                    "dilim": "4h", "strateji": "ict_long", "yon": "LONG",
                    "basarili": True, "sonuc": {"orderId": 1}}],
                  open(self.kayit_dosya, "w"))
        ozet = tn.esle({"BTCUSDT": 0.14})
        self.assertEqual(api.emirler(), [])
        self.assertEqual(ozet["atlanan"], 1)
        self.assertEqual(self._kayitlar()[-1]["sonuc"]["atlandi"], "son 24s ICT emri var")

    def test_eski_ict_emri_engellemez(self):
        """48 saat oncesinin ICT kaydi pencerenin disinda -> emir gider."""
        api = self._api(poz=0.0)
        eski = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 48 * 3600))
        json.dump([{"zaman": eski, "sembol": "BTCUSDT", "strateji": "ict_long",
                    "basarili": True, "sonuc": {}}], open(self.kayit_dosya, "w"))
        tn.esle({"BTCUSDT": 0.14})
        self.assertEqual(len(api.emirler()), 1)

    def test_kendi_trend_kaydi_engellemez(self):
        """Dunku kendi TREND kaydimiz ICT sanilmamali."""
        api = self._api(poz=0.0)
        json.dump([{"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "tur": "TREND",
                    "sembol": "BTCUSDT", "basarili": True, "sonuc": {}}],
                  open(self.kayit_dosya, "w"))
        tn.esle({"BTCUSDT": 0.14})
        self.assertEqual(len(api.emirler()), 1)

    # ---------------- 5) kuru mod ----------------
    def test_kuru_mod_yazmaz(self):
        api = self._api(poz=0.0)
        ozet = tn.esle({"BTCUSDT": 0.14}, kuru=True)
        self.assertTrue(ozet["kuru"])
        self.assertEqual(ozet["emir"], 1)                      # planlandi
        self.assertEqual(api.emirler(), [])                    # ama gonderilmedi
        self.assertFalse(os.path.exists(self.kayit_dosya))     # dosyaya da yazilmadi

    def test_anahtar_yoksa_otomatik_kuru(self):
        api = self._api(poz=0.0)
        tn.KEY = tn.SECRET = ""
        ozet = tn.esle({"BTCUSDT": 0.14})
        self.assertTrue(ozet["kuru"])
        self.assertEqual(api.emirler(), [])
        # imzali hicbir istek atilmamali (positionRisk dahil)
        self.assertFalse([1 for m, yol, p in api.cagrilar if "signature" in p])

    def test_coklu_sembol_ozeti(self):
        api = self._api(poz=0.0)
        ozet = tn.esle({"BTCUSDT": 0.14, "ETHUSDT": 0.0})      # ETH agirlik 0 -> atla
        self.assertEqual(ozet["emir"], 1)
        self.assertEqual(ozet["atlanan"], 1)
        self.assertEqual(len(api.emirler()), 1)

    # ---------------- 6) sahiplik: ICT / sahipsiz pozisyona dokunma ----------------
    def test_hesaptaki_ict_pozisyonuna_dokunmaz(self):
        """Canli olay 2026-09-13: hesapta 4.19 adet ICT kalintisi varken trend hedefi 0.2.
        Eski kod SATIYORDU; defterle trend sadece kendi payini ALMALI (test: hedef 0.5 > min notional)."""
        api = self._api(poz=4.19)
        ozet = tn.esle({"BTCUSDT": 0.05}, defter={})          # 1000*0.05/100 = 0.5 (50 USDT)
        e = api.emirler()
        self.assertEqual(len(e), 1, e)
        self.assertEqual(e[0]["side"], "BUY")
        self.assertAlmostEqual(float(e[0]["quantity"]), 0.5, places=6)
        self.assertNotIn("reduceOnly", e[0])
        self.assertAlmostEqual(ozet["defter"]["BTCUSDT"], 0.5, places=6)

    def test_screener_acik_pozisyonu_varsa_atla(self):
        api = self._api(poz=0.0)
        json.dump([{"symbol": "BTCUSDT", "status": "open", "sinyal_gonderildi": True}],
                  open(tn.POZ_DOSYA, "w"))
        ozet = tn.esle({"BTCUSDT": 0.14}, defter={})
        self.assertEqual(api.emirler(), [])
        self.assertEqual(ozet["atlanan"], 1)
        self.assertEqual(self._kayitlar()[-1]["sonuc"]["atlandi"], "screener acik pozisyonu var")

    def test_sessiz_screener_pozisyonu_engellemez(self):
        """Testnete gonderilmemis (sessiz dilim) pozisyon hesapta yok -> trend calisir."""
        api = self._api(poz=0.0)
        json.dump([{"symbol": "BTCUSDT", "status": "pending", "sinyal_gonderildi": False}],
                  open(tn.POZ_DOSYA, "w"))
        tn.esle({"BTCUSDT": 0.14}, defter={})
        self.assertEqual(len(api.emirler()), 1)

    def test_defter_trend_kayitlarindan_kurulur(self):
        """defter verilmezse son basarili TREND kaydinin hedef_miktar'i kullanilir."""
        api = self._api(poz=5.6)                              # hesapta 4.2 ICT + 1.4 trend
        json.dump([{"zaman": "2026-09-13T01:36:20", "tur": "TREND", "sembol": "BTCUSDT",
                    "basarili": True, "sonuc": {"hedef_miktar": 1.4}}],
                  open(self.kayit_dosya, "w"))
        ozet = tn.esle({"BTCUSDT": 0.14})                     # hedef 1.4 = defter 1.4
        self.assertEqual(api.emirler(), [])
        self.assertAlmostEqual(ozet["defter"]["BTCUSDT"], 1.4, places=6)

    def test_reduce_only_hesap_yetersizse_konmaz(self):
        """Defter 1.4 ama hesapta sadece 0.5 (dis mudahale): 0.9 satis reduceOnly ile reddedilirdi."""
        api = self._api(poz=0.5)
        tn.esle({"BTCUSDT": 0.05}, defter={"BTCUSDT": 1.4})
        e = api.emirler()
        self.assertEqual(len(e), 1, e)
        self.assertEqual(e[0]["side"], "SELL")
        self.assertNotIn("reduceOnly", e[0])

    def test_kuru_mod_defteri_degistirmez(self):
        self._api(poz=0.0)
        d = {"BTCUSDT": 0.0}
        tn.esle({"BTCUSDT": 0.14}, kuru=True, defter=d)
        self.assertEqual(d, {"BTCUSDT": 0.0})

    # ---------------- 7) maliyet defteri ve demo K/Z raporu ----------------
    def test_maliyet_alimda_ortalama(self):
        self._api(poz=0.0, son_fiyat=100.0)
        ozet = tn.esle({"BTCUSDT": 0.14}, defter={}, maliyet={})
        self.assertAlmostEqual(ozet["maliyet"]["BTCUSDT"]["ort"], 100.0, places=6)
        self.assertAlmostEqual(ozet["maliyet"]["BTCUSDT"]["gerceklesen"], 0.0, places=6)

    def test_maliyet_ekleme_ortalamayi_gunceller(self):
        self._api(poz=1.0, son_fiyat=100.0)                   # 1.0 @ 80 elde, 0.4 @ 100 ekle
        ozet = tn.esle({"BTCUSDT": 0.14}, defter={"BTCUSDT": 1.0},
                       maliyet={"BTCUSDT": {"ort": 80.0, "gerceklesen": 0.0}})
        self.assertAlmostEqual(ozet["maliyet"]["BTCUSDT"]["ort"], (80 * 1.0 + 100 * 0.4) / 1.4, places=6)

    def test_maliyet_satista_gerceklesen(self):
        self._api(poz=1.4, son_fiyat=100.0)                   # 1.4 @ 80 -> 0.5, 0.9 satilir
        ozet = tn.esle({"BTCUSDT": 0.05}, defter={"BTCUSDT": 1.4},
                       maliyet={"BTCUSDT": {"ort": 80.0, "gerceklesen": 0.0}})
        m = ozet["maliyet"]["BTCUSDT"]
        self.assertAlmostEqual(m["gerceklesen"], 18.0, places=6)     # (100-80) x 0.9
        self.assertAlmostEqual(m["ort"], 80.0, places=6)

    def test_maliyet_kur_trend_kayitlarindan(self):
        """Canli 2026-09-13: BNB 0.2 @ 729.13 (ICT kalintisindan devralindi), sonra 0.1'e indi @ 800."""
        json.dump([{"zaman": "2026-09-13T01:36:19", "tur": "TREND", "sembol": "BNBUSDT", "basarili": True,
                    "sonuc": {"hedef_miktar": 0.2, "fiyat": 729.13}},
                   {"zaman": "2026-09-14T01:36:19", "tur": "TREND", "sembol": "BNBUSDT", "basarili": True,
                    "sonuc": {"hedef_miktar": 0.1, "fiyat": 800.0}},
                   {"zaman": "2026-09-14T01:36:20", "tur": "TREND", "sembol": "ETHUSDT", "basarili": None,
                    "sonuc": {"atlandi": "min notional"}}], open(self.kayit_dosya, "w"))
        m = tn.maliyet_kur()
        self.assertAlmostEqual(m["BNBUSDT"]["ort"], 729.13, places=6)
        self.assertAlmostEqual(m["BNBUSDT"]["gerceklesen"], (800.0 - 729.13) * 0.1, places=6)
        self.assertNotIn("ETHUSDT", m)

    def test_kuru_mod_maliyeti_degistirmez(self):
        self._api(poz=0.0)
        m = {}
        tn.esle({"BTCUSDT": 0.14}, kuru=True, defter={}, maliyet=m)
        self.assertEqual(m, {})

    def test_pnl_raporu(self):
        self._api(poz=1.4, son_fiyat=100.0)
        rap = tn.pnl_raporu({"BTCUSDT": 1.4}, {"BTCUSDT": {"ort": 80.0, "gerceklesen": 5.0}})
        self.assertAlmostEqual(rap["deger"], 140.0, places=6)
        self.assertAlmostEqual(rap["acik_kz"], 28.0, places=6)       # (100-80) x 1.4
        self.assertAlmostEqual(rap["toplam_kz"], 33.0, places=6)
        self.assertAlmostEqual(rap["getiri"], 0.033, places=6)       # 33 / 1000
        self.assertEqual(len(rap["satirlar"]), 1)
        self.assertNotIn("<", rap["satirlar"][0])                    # Telegram HTML guvenligi
        self.assertNotIn(">", rap["satirlar"][0])

    def test_pnl_raporu_anahtarsiz_none(self):
        self._api(poz=0.0)
        tn.KEY = tn.SECRET = ""
        self.assertIsNone(tn.pnl_raporu({"BTCUSDT": 1.0}, {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
