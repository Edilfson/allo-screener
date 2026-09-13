"""
BINANCE FUTURES TESTNET ISLEMCISI
screener.py sinyal urettiginde testnet e gercek limit emir acar.
positions.json takibi AYNEN devam eder - bu EK bir katman.
NOT: Binance Aralik 2025 degisikligi - kosullu emirler (STOP_MARKET,
TAKE_PROFIT_MARKET) artik /fapi/v1/algoOrder uc noktasindan gonderilir.
Eski /fapi/v1/order -4120 hatasi veriyordu.

UYARI: Testnet likiditesi gercek piyasadan farklidir; emirler kolay dolar,
kayma gerceklci degildir. Sonuclari karsilastirma amacli kullan.
Secret: TESTNET_API_KEY, TESTNET_API_SECRET
Anahtar: https://testnet.binancefuture.com -> API Key
"""

import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlencode

import requests

BASE = "https://testnet.binancefuture.com"
KEY = os.environ.get("TESTNET_API_KEY", "")
SECRET = os.environ.get("TESTNET_API_SECRET", "")
STATE = "testnet_orders.json"

RISK_USDT = float(os.environ.get("TESTNET_RISK_USDT", "50"))
LEVERAGE = int(os.environ.get("TESTNET_LEVERAGE", "5"))


def _imzali(yol, params=None, method="GET"):
    if not KEY or not SECRET:
        return False, {"hata": "anahtar yok"}
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 10000
    q = urlencode(p)
    sig = hmac.new(SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}{yol}?{q}&signature={sig}"
    try:
        r = requests.request(method, url, headers={"X-MBX-APIKEY": KEY}, timeout=20)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"hata": str(e)}


def bakiye():
    ok, c = _imzali("/fapi/v2/balance")
    if not ok:
        return None
    for x in c:
        if x.get("asset") == "USDT":
            return float(x.get("balance", 0))
    return None


def sembol_bilgi(sym):
    """tick, adim, miktar hassasiyeti, fiyat hassasiyeti, MAX miktar.
    MAX miktar = min(LOT_SIZE.maxQty, MARKET_LOT_SIZE.maxQty): stop/TP piyasa
    emri oldugu icin MARKET siniri belirleyicidir (canli: 16 sinyal -4005 ile red)."""
    try:
        r = requests.get(f"{BASE}/fapi/v1/exchangeInfo", timeout=20)
        for s in r.json().get("symbols", []):
            if s["symbol"] == sym:
                tick = adim = None
                maxq = []
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                    if f["filterType"] == "LOT_SIZE":
                        adim = float(f["stepSize"])
                        maxq.append(float(f.get("maxQty", 0) or 0))
                    if f["filterType"] == "MARKET_LOT_SIZE":
                        maxq.append(float(f.get("maxQty", 0) or 0))
                maxq = [m for m in maxq if m > 0]
                return (tick, adim, s.get("quantityPrecision", 3),
                        s.get("pricePrecision", 2), min(maxq) if maxq else None)
    except Exception:
        pass
    return None, None, 3, 2, None


def kullanilabilir_bakiye():
    ok, c = _imzali("/fapi/v2/balance")
    if not ok:
        return None
    for x in c:
        if x.get("asset") == "USDT":
            return float(x.get("availableBalance", x.get("balance", 0)))
    return None


def _yuvarla(deger, adim, hassas):
    if adim:
        deger = round(deger / adim) * adim
    return round(deger, hassas)

def emir_ac(plan, sembol):
    """Limit giris + stop + TP emirlerini testnet te acar."""
    side = plan["side"]
    yon = "BUY" if side == 1 else "SELL"
    ters = "SELL" if side == 1 else "BUY"
    tick, adim, qp, pp, max_miktar = sembol_bilgi(sembol)

    giris = _yuvarla(plan["entry"], tick, pp)
    stop = _yuvarla(plan["stop"], tick, pp)
    tp = _yuvarla(plan["tps"][0]["p"], tick, pp)

    risk_birim = abs(giris - stop)
    if risk_birim <= 0:
        return False, {"hata": "risk sifir"}
    miktar = _yuvarla(RISK_USDT / risk_birim, adim, qp)
    if miktar <= 0:
        return False, {"hata": "miktar sifir"}

    # MIKTAR SINIRI: borsanin max miktarini asma (asarsa stop emri reddedilir,
    # giris iptal olur -> sinyal bosa gider). Kisilirsa risk de kisilir; kaydedilir.
    kisildi = False
    if max_miktar and miktar > max_miktar:
        miktar = _yuvarla(max_miktar, adim, qp)
        kisildi = True

    # MARJIN ON KONTROLU: dar stoplarda nominal 5-8k USDT oluyor, 5x kaldiracla
    # marjin 1-1.7k; eszamanli 8 emirde bakiye yetmiyordu (12x "Margin is insufficient")
    gerekli_marjin = miktar * giris / LEVERAGE * 1.05
    bakiye_k = kullanilabilir_bakiye()
    if bakiye_k is not None and bakiye_k < gerekli_marjin:
        return False, {"hata": "marjin yetersiz (on kontrol)",
                       "gerekli": round(gerekli_marjin, 1), "kullanilabilir": round(bakiye_k, 1)}

    _imzali("/fapi/v1/leverage", {"symbol": sembol, "leverage": LEVERAGE}, "POST")

    ok, giris_emri = _imzali("/fapi/v1/order", {
        "symbol": sembol, "side": yon, "type": "LIMIT", "timeInForce": "GTC",
        "quantity": miktar, "price": giris}, "POST")
    if not ok:
        return False, giris_emri

    # KORUMA EMIRLERI - sonuclari MUTLAKA kontrol edilir
    # (onceki surumde sonuc yok sayiliyordu; stop reddedilse bile
    #  "emir acildi" deniyordu -> pozisyon korumasiz kaliyordu)
    ok_stop, c_stop = _imzali("/fapi/v1/algoOrder", {
        "symbol": sembol, "side": ters, "algoType": "CONDITIONAL", "type": "STOP_MARKET", "orderType": "STOP_MARKET",
        "triggerPrice": stop, "quantity": miktar, "reduceOnly": "true",
        "workingType": "MARK_PRICE"}, "POST")
    ok_tp, c_tp = _imzali("/fapi/v1/algoOrder", {
        "symbol": sembol, "side": ters, "algoType": "CONDITIONAL", "type": "TAKE_PROFIT_MARKET", "orderType": "TAKE_PROFIT_MARKET",
        "triggerPrice": tp, "quantity": miktar, "reduceOnly": "true",
        "workingType": "MARK_PRICE"}, "POST")

    if not ok_stop:
        print(f"  [!] {sembol} STOP EMRI REDDEDILDI: {c_stop}")
    if not ok_tp:
        print(f"  [!] {sembol} TP EMRI REDDEDILDI: {c_tp}")

    # STOP yoksa pozisyon KORUMASIZ - giris emrini iptal et
    if not ok_stop:
        _imzali("/fapi/v1/allOpenOrders", {"symbol": sembol}, "DELETE")
        return False, {"hata": "stop emri reddedildi, giris iptal edildi",
                       "stop_hatasi": c_stop}

    return True, {"orderId": giris_emri.get("orderId"), "miktar": miktar,
                  "giris": giris, "stop": stop, "tp": tp,
                  "risk_usdt": round(miktar * risk_birim, 2), "miktar_kisildi": kisildi,
                  "stop_ok": ok_stop, "tp_ok": ok_tp}


def kaydet(kayit):
    try:
        L = json.load(open(STATE)) if os.path.exists(STATE) else []
    except Exception:
        L = []
    L.append(kayit)
    json.dump(L[-2000:], open(STATE, "w"), indent=1, default=str)


def sinyali_isle(plan, sembol, dilim, strateji):
    """screener.py buradan cagirir."""
    if not KEY or not SECRET:
        print("  [testnet] anahtar yok, atlandi")
        return None, {"hata": "anahtar yok"}
    ok, sonuc = emir_ac(plan, sembol)
    kayit = {"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "sembol": sembol,
             "dilim": dilim, "strateji": strateji,
             "yon": "LONG" if plan["side"] == 1 else "SHORT",
             "basarili": ok, "sonuc": sonuc}
    kaydet(kayit)
    print(f"  [testnet] {sembol} {dilim}: {'emir acildi' if ok else 'HATA'} {sonuc}")
    return ok, sonuc


def acik_emirler(sembol=None):
    """Bekleyen emirler: NORMAL + ALGO (kosullu) emirler birlikte.
    Binance Aralik 2025'ten sonra stop/TP emirleri ayri listede tutuluyor."""
    p = {"symbol": sembol} if sembol else {}
    liste = []
    ok, c = _imzali("/fapi/v1/openOrders", p)
    if ok and isinstance(c, list):
        liste += c
    # ALGO (kosullu) emirler ayri uc noktada
    ok2, c2 = _imzali("/fapi/v1/openAlgoOrders", p)
    if ok2 and isinstance(c2, list):
        for a in c2:
            a["_algo"] = True
            a.setdefault("type", a.get("orderType", "ALGO"))
            a.setdefault("stopPrice", a.get("triggerPrice"))
        liste += c2
    return liste


def emirleri_iptal(sembol):
    """Bir sembolun TUM bekleyen emirlerini iptal eder: normal + algo (stop/tp)."""
    ok, c = _imzali("/fapi/v1/allOpenOrders", {"symbol": sembol}, "DELETE")
    ok2, c2 = _imzali("/fapi/v1/algoOpenOrders", {"symbol": sembol}, "DELETE")
    return (ok or ok2), {"normal": c, "algo": c2}


def pozisyon_var_mi(sembol):
    """Testnet'te o sembolde ACIK pozisyon var mi? (limit doldu mu)"""
    ok, c = _imzali("/fapi/v2/positionRisk", {"symbol": sembol})
    if not ok:
        return None
    for x in (c if isinstance(c, list) else []):
        if x.get("symbol") == sembol:
            miktar = float(x.get("positionAmt", 0))
            if abs(miktar) > 0:
                return {"miktar": miktar,
                        "giris": float(x.get("entryPrice", 0)),
                        "kar": float(x.get("unRealizedProfit", 0))}
    return None


TREND_STATE = "trend_state.json"   # trend portfoyunun testnet defteri burada
MIN_KAPAT_USDT = 5.0               # bunun altindaki kalinti kapatilmaz (toz)


def pozisyon_miktari(sembol):
    """Net positionAmt (hedge modda bacaklar toplanir). Pozisyon yoksa 0.0, OKUNAMAZSA None.
    (pozisyon_var_mi hata ile 'yok'u ayirt edemiyordu -> iptal_et stop emrini de siliyordu)"""
    ok, c = _imzali("/fapi/v2/positionRisk", {"symbol": sembol})
    if not ok or not isinstance(c, list):
        return None
    toplam = 0.0
    for x in c:
        if x.get("symbol") == sembol:
            try:
                toplam += float(x.get("positionAmt", 0) or 0)
            except Exception:
                pass
    return toplam


def _fiyat(sembol):
    try:
        r = requests.get(f"{BASE}/fapi/v1/ticker/price", params={"symbol": sembol}, timeout=20)
        return float(r.json()["price"])
    except Exception:
        return None


def trend_payi(sembol):
    """Trend portfoyunun bu sembolde tuttugu miktar (trend_state.json portfoy.testnet_defter;
    yoksa TREND kayitlarindan kurulur). Okunamazsa 0."""
    try:
        st = json.load(open(TREND_STATE)) if os.path.exists(TREND_STATE) else {}
        d = (st.get("portfoy") or {}).get("testnet_defter")
        if d is None:
            import trend_testnet          # gec import: dongusel importu onler
            d = trend_testnet.defter_kur()
        return float((d or {}).get(sembol, 0.0) or 0.0)
    except Exception:
        return 0.0


def ict_giris_miktari(sembol):
    """Son BASARILI ICT giris kaydinin net miktari (+long/-short); kayit yoksa None."""
    try:
        L = json.load(open(STATE)) if os.path.exists(STATE) else []
    except Exception:
        return None
    for r in reversed(L):
        if (isinstance(r, dict) and r.get("sembol") == sembol and r.get("strateji")
                and r.get("basarili") and not r.get("tur")):
            try:
                m = float((r.get("sonuc") or {}).get("miktar") or 0)
            except Exception:
                return None
            return m if r.get("yon") == "LONG" else -m
    return None


def piyasa_kapat(sembol, net):
    """net (+long/-short) kadar pozisyonu MARKET reduceOnly ile kapatir; max miktar sinirinda boler."""
    _t, adim, qp, _pp, maxq = sembol_bilgi(sembol)
    kalan = _yuvarla(abs(net), adim, qp)
    yanitlar, hepsi_ok = [], True
    for _ in range(10):
        if kalan <= 0:
            break
        q = min(kalan, maxq) if maxq else kalan
        q = _yuvarla(q, adim, qp)
        ok, c = _imzali("/fapi/v1/order", {"symbol": sembol, "side": "SELL" if net > 0 else "BUY",
                                           "type": "MARKET", "quantity": q, "reduceOnly": "true"}, "POST")
        yanitlar.append(c); hepsi_ok = hepsi_ok and ok
        if not ok:
            break
        kalan = _yuvarla(kalan - q, adim, qp)
    return hepsi_ok, yanitlar


def ict_payini_kapat(sembol, poz, ict_net=None, trend=None, kuru=False):
    """Hesaptaki trend DISI payi kapatir. ict_net verilirse (ICT giris miktari) en fazla o kadar.
    Return None (kapatilacak bir sey yok) veya {kapat, trend_payi, ok, yanit}."""
    trend = trend_payi(sembol) if trend is None else trend
    hedef = poz - trend
    if ict_net is not None:
        hedef = (1 if hedef > 0 else -1) * min(abs(hedef), abs(ict_net)) if hedef * ict_net > 0 else 0.0
    if hedef * poz <= 0:
        return None
    q = (1 if poz > 0 else -1) * min(abs(hedef), abs(poz))
    p = _fiyat(sembol)
    if p is not None and abs(q) * p < MIN_KAPAT_USDT:
        return None
    if kuru:
        return {"kapat": q, "trend_payi": trend, "kuru": True}
    ok, c = piyasa_kapat(sembol, q)
    return {"kapat": q, "trend_payi": trend, "ok": ok, "yanit": c}


def iptal_et(sembol, sebep=""):
    """screener.py bir ICT pozisyonunu kapattiginda (stop/TP/iptal/zaman asimi/temizlik) cagirir.
    Testneti teorik kayitla esler:
      1) pozisyon OKUNAMAZSA hicbir seye dokunmaz (onceki surum API hatasini 'pozisyon yok' sanip
         stop dahil tum emirleri siliyordu -> 30 Agustos BNB: 2 hafta stopsuz 4.19 BNB kaldi)
      2) bekleyen emirleri (giris limiti + stop/TP) iptal eder
      3) hesapta ICT payi kaldiysa MARKET reduceOnly ile kapatir; miktar = min(ICT giris miktari,
         hesap - trend payi) -> trend portfoyunun payina dokunulmaz"""
    if not KEY or not SECRET:
        return
    zaman = time.strftime("%Y-%m-%dT%H:%M:%S")
    poz = pozisyon_miktari(sembol)
    if poz is None:
        kaydet({"zaman": zaman, "sembol": sembol, "tur": "IPTAL", "sebep": sebep, "basarili": False,
                "sonuc": {"hata": "pozisyon okunamadi, hicbir emre dokunulmadi"}})
        print(f"  [testnet] {sembol}: pozisyon OKUNAMADI, emirlere dokunulmadi ({sebep})")
        return
    ok, c = emirleri_iptal(sembol)
    kapanis = ict_payini_kapat(sembol, poz, ict_net=ict_giris_miktari(sembol)) if poz else None
    basarili = bool(ok) and (kapanis is None or bool(kapanis.get("ok")))
    kaydet({"zaman": zaman, "sembol": sembol, "tur": "IPTAL", "sebep": sebep, "basarili": basarili,
            "sonuc": c, "pozisyon": poz, "kapanis": kapanis})
    ek = f", ICT payi {kapanis['kapat']} kapatildi" if kapanis else ""
    print(f"  [testnet] {sembol}: emirler iptal edildi{ek} ({sebep})"
          if basarili else f"  [testnet] {sembol}: iptal/kapanis HATASI {c} {kapanis}")


if __name__ == "__main__":
    b = bakiye()
    print("Testnet baglantisi:", "OK" if b is not None else "BASARISIZ")
    if b is not None:
        print(f"Bakiye: {b:,.2f} USDT | risk: {RISK_USDT} USDT | kaldirac: {LEVERAGE}x")
        ae = acik_emirler()
        print(f"Bekleyen emir sayisi: {len(ae)}")
        for e in ae[:10]:
            print(f"  {e.get('symbol')} {e.get('side')} {e.get('type')} @ {e.get('price')}")
