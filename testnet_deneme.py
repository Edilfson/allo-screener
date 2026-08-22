"""TESTNET DENEME ARACI - Actions -> Testnet Test ile elle calistirilir.
  durum  : sadece rapor | market : hemen ac | limit : beklesin
"""

import os
import time

import requests

import testnet_trader as T

sym = os.environ.get("SEMBOL", "BTCUSDT").upper()
yon = os.environ.get("YON", "LONG").upper()
mod = os.environ.get("MOD", "durum").lower()
side = 1 if yon == "LONG" else -1

print("=" * 60)
b = T.bakiye()
print("1) BAGLANTI:", "OK" if b is not None else "BASARISIZ")
if b is None:
    print("   API anahtarlarini kontrol et.")
    raise SystemExit(1)
print(f"   Bakiye: {b:,.2f} USDT")

fiyat = float(requests.get(f"{T.BASE}/fapi/v1/ticker/price",
                           params={"symbol": sym}, timeout=20).json()["price"])
print(f"2) {sym} fiyat: {fiyat}")

tick, adim, qp, pp = T.sembol_bilgi(sym)

if mod == "durum":
    print("3) SADECE DURUM RAPORU - emir acilmayacak")

elif mod == "market":
    stop = T._yuvarla(fiyat * (0.98 if side == 1 else 1.02), tick, pp)
    tp = T._yuvarla(fiyat + side * 5 * abs(fiyat - stop), tick, pp)
    miktar = T._yuvarla(T.RISK_USDT / abs(fiyat - stop), adim, qp)
    print(f"3) PIYASA EMRI: {yon} miktar={miktar} stop={stop} tp={tp}")
    T._imzali("/fapi/v1/leverage", {"symbol": sym, "leverage": T.LEVERAGE}, "POST")
    ok, c = T._imzali("/fapi/v1/order", {
        "symbol": sym, "side": "BUY" if side == 1 else "SELL",
        "type": "MARKET", "quantity": miktar}, "POST")
    print("   giris:", "ACILDI" if ok else "HATA", "" if ok else c)
    if ok:
        ters = "SELL" if side == 1 else "BUY"
        ok2, c2 = T._imzali("/fapi/v1/algoOrder", {
            "symbol": sym, "side": ters, "algoType": "CONDITIONAL", "orderType": "STOP_MARKET",
            "triggerPrice": stop, "quantity": miktar, "reduceOnly": "true",
            "workingType": "MARK_PRICE"}, "POST")
        print("   STOP:", "OK" if ok2 else "REDDEDILDI", "" if ok2 else c2)
        ok3, c3 = T._imzali("/fapi/v1/algoOrder", {
            "symbol": sym, "side": ters, "algoType": "CONDITIONAL", "orderType": "TAKE_PROFIT_MARKET",
            "triggerPrice": tp, "quantity": miktar, "reduceOnly": "true",
            "workingType": "MARK_PRICE"}, "POST")
        print("   TP  :", "OK" if ok3 else "REDDEDILDI", "" if ok3 else c3)
        if not ok2:
            print("   [!] STOP YOK - pozisyon KORUMASIZ, elle kapat!")

else:
    giris = fiyat * (0.97 if side == 1 else 1.03)
    stop = giris * (0.98 if side == 1 else 1.02)
    tp = giris + side * 5 * abs(giris - stop)
    plan = {"side": side, "entry": giris, "stop": stop,
            "tps": [{"p": tp, "w": 1.0, "r": 5.0, "hit": False}]}
    ok, c = T.emir_ac(plan, sym)
    print("3) LIMIT EMIR:", "ACILDI" if ok else "HATA")
    print("  ", c)

time.sleep(3)
print("4) ACIK POZISYON:")
p = T.pozisyon_var_mi(sym)
print("   ", p if p else "yok")

ae = T.acik_emirler(sym)
print(f"5) BEKLEYEN EMIRLER ({len(ae)}):")
for e in ae:
    print(f"   {e.get(chr(39)+chr(116)+chr(121)+chr(112)+chr(101)+chr(39))} {e.get(chr(39)+chr(115)+chr(105)+chr(100)+chr(101)+chr(39))} @ {e.get(chr(39)+chr(115)+chr(116)+chr(111)+chr(112)+chr(80)+chr(114)+chr(105)+chr(99)+chr(101)+chr(39)) or e.get(chr(39)+chr(112)+chr(114)+chr(105)+chr(99)+chr(101)+chr(39))}")
print("=" * 60)
