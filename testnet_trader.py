"""
BINANCE FUTURES TESTNET ISLEMCISI
screener.py sinyal urettiginde testnet e gercek limit emir acar.
positions.json takibi AYNEN devam eder - bu EK bir katman.
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
    try:
        r = requests.get(f"{BASE}/fapi/v1/exchangeInfo", timeout=20)
        for s in r.json().get("symbols", []):
            if s["symbol"] == sym:
                tick = adim = None
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                    if f["filterType"] == "LOT_SIZE":
                        adim = float(f["stepSize"])
                return tick, adim, s.get("quantityPrecision", 3), s.get("pricePrecision", 2)
    except Exception:
        pass
    return None, None, 3, 2


def _yuvarla(deger, adim, hassas):
    if adim:
        deger = round(deger / adim) * adim
    return round(deger, hassas)

def emir_ac(plan, sembol):
    """Limit giris + stop + TP emirlerini testnet te acar."""
    side = plan["side"]
    yon = "BUY" if side == 1 else "SELL"
    ters = "SELL" if side == 1 else "BUY"
    tick, adim, qp, pp = sembol_bilgi(sembol)

    giris = _yuvarla(plan["entry"], tick, pp)
    stop = _yuvarla(plan["stop"], tick, pp)
    tp = _yuvarla(plan["tps"][0]["p"], tick, pp)

    risk_birim = abs(giris - stop)
    if risk_birim <= 0:
        return False, {"hata": "risk sifir"}
    miktar = _yuvarla(RISK_USDT / risk_birim, adim, qp)
    if miktar <= 0:
        return False, {"hata": "miktar sifir"}

    _imzali("/fapi/v1/leverage", {"symbol": sembol, "leverage": LEVERAGE}, "POST")

    ok, giris_emri = _imzali("/fapi/v1/order", {
        "symbol": sembol, "side": yon, "type": "LIMIT", "timeInForce": "GTC",
        "quantity": miktar, "price": giris}, "POST")
    if not ok:
        return False, giris_emri

    _imzali("/fapi/v1/order", {
        "symbol": sembol, "side": ters, "type": "STOP_MARKET",
        "stopPrice": stop, "closePosition": "true"}, "POST")
    _imzali("/fapi/v1/order", {
        "symbol": sembol, "side": ters, "type": "TAKE_PROFIT_MARKET",
        "stopPrice": tp, "closePosition": "true"}, "POST")

    return True, {"orderId": giris_emri.get("orderId"), "miktar": miktar,
                  "giris": giris, "stop": stop, "tp": tp}


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
        return
    ok, sonuc = emir_ac(plan, sembol)
    kayit = {"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "sembol": sembol,
             "dilim": dilim, "strateji": strateji,
             "yon": "LONG" if plan["side"] == 1 else "SHORT",
             "basarili": ok, "sonuc": sonuc}
    kaydet(kayit)
    print(f"  [testnet] {sembol} {dilim}: {'emir acildi' if ok else 'HATA'} {sonuc}")


if __name__ == "__main__":
    b = bakiye()
    print("Testnet baglantisi:", "OK" if b is not None else "BASARISIZ")
    if b is not None:
        print(f"Bakiye: {b:,.2f} USDT | risk: {RISK_USDT} USDT | kaldirac: {LEVERAGE}x")
