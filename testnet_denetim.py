"""
TESTNET DENETIMI - sahipsiz pozisyon ve bayat emir temizligi
============================================================
Hesaptaki her pozisyon/emir sahibine gore siniflanir:
  ICT aktif   : positions.json'da testnete gonderilmis acik/bekleyen ICT pozisyonu var -> DOKUNULMAZ
                (stop emri yoksa UYARI verilir)
  trend payi  : hesap miktari == trend defteri (trend_state.json portfoy.testnet_defter) -> DOKUNULMAZ
  SAHIPSIZ    : hesap - trend payi, sahibi olmayan kalinti -> --kapat ile emirler iptal + MARKET reduceOnly
  bayat emir  : pozisyonu ve sahibi olmayan bekleyen emir (dolarsa yeni sahipsiz yaratir) -> --kapat ile iptal
YAS KURALI: son YAS_SAAT saatte guncellenmis pozisyon/emire dokunulmaz (o anda calisan screener veya
trend kosusunun yeni actigi islem henuz state dosyalarina yazilmamis olabilir). Sahipsiz kalinti eskidir.
Rapor Telegram TOPIC_SUMMARY'ye gider. Anahtar yoksa hicbir sey yapmaz.
Calistirma: python testnet_denetim.py [--kapat]
"""
import argparse
import json
import os
import time

import requests

import testnet_trader as tt
import trend_testnet as tn

YAS_SAAT = float(os.environ.get("DENETIM_YAS_SAAT", "6"))
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC = os.environ.get("TOPIC_SUMMARY")


def _yas_saat(ms):
    try:
        ms = float(ms or 0)
    except Exception:
        return None
    return None if ms <= 0 else (time.time() * 1000 - ms) / 3_600_000


def tum_pozisyonlar():
    """{sembol: {miktar, mark, guncelleme_ms}} (sadece sifir olmayanlar). Okunamazsa None."""
    ok, c = tt._imzali("/fapi/v2/positionRisk")
    if not ok or not isinstance(c, list):
        return None
    out = {}
    for x in c:
        try:
            m = float(x.get("positionAmt", 0) or 0)
        except Exception:
            continue
        if m == 0:
            continue
        s = x["symbol"]
        o = out.setdefault(s, {"miktar": 0.0, "mark": 0.0, "guncelleme_ms": 0})
        o["miktar"] += m
        o["mark"] = float(x.get("markPrice", 0) or 0) or o["mark"]
        o["guncelleme_ms"] = max(o["guncelleme_ms"], int(float(x.get("updateTime", 0) or 0)))
    return out


def tum_emirler():
    """{sembol: [emir]} normal + algo. Return (dict, okundu_mu)."""
    out, okundu = {}, True
    for yol, algo in (("/fapi/v1/openOrders", False), ("/fapi/v1/openAlgoOrders", True)):
        ok, c = tt._imzali(yol)
        if not ok or not isinstance(c, list):
            okundu = False
            continue
        for e in c:
            e["_algo"] = algo
            out.setdefault(e.get("symbol"), []).append(e)
    return out, okundu


def _emir_zamani(e):
    return max(int(float(e.get(k, 0) or 0)) for k in ("updateTime", "time", "createTime"))


def _stop_var(emirler):
    return any("STOP" in str(e.get("orderType") or e.get("type") or "").upper() for e in emirler)


def denetle(kapat=False):
    ozet = {"kapat": kapat, "satirlar": [], "islem": 0, "uyari": 0, "hata": 0}
    if not tt.KEY or not tt.SECRET:
        ozet["satirlar"].append("anahtar yok - denetim atlandi")
        return ozet
    pozlar = tum_pozisyonlar()
    if pozlar is None:
        ozet["hata"] += 1
        ozet["satirlar"].append("pozisyonlar OKUNAMADI - hicbir seye dokunulmadi")
        return ozet
    emirler, emir_ok = tum_emirler()
    if not emir_ok:
        ozet["uyari"] += 1
        ozet["satirlar"].append("emir listesi tam okunamadi (bayat emir temizligi eksik olabilir)")

    for s in sorted(set(pozlar) | set(k for k in emirler if k)):
        p = pozlar.get(s, {"miktar": 0.0, "mark": 0.0, "guncelleme_ms": 0})
        amt, e = p["miktar"], emirler.get(s, [])
        trend = tt.trend_payi(s)
        fiyat = p["mark"] or tt._fiyat(s) or 0.0

        if tn.screener_pozisyonu_var(s):
            if amt != 0 and abs(amt - trend) * fiyat >= tt.MIN_KAPAT_USDT and not _stop_var(e):
                ozet["uyari"] += 1
                ozet["satirlar"].append(f"UYARI {s}: ICT aktif, hesap {amt} ama STOP emri yok")
            else:
                ozet["satirlar"].append(f"{s}: ICT aktif (hesap {amt}, trend {trend}) - dokunulmadi")
            continue

        fazla = amt - trend
        if amt != 0 and fazla * amt > 0 and abs(fazla) * fiyat >= tt.MIN_KAPAT_USDT:
            yas = _yas_saat(p["guncelleme_ms"])
            if yas is None or yas < YAS_SAAT:
                ozet["satirlar"].append(f"{s}: sahipsiz gorunen {fazla:+g} ama yeni ({yas and round(yas,1)} sa) - beklendi")
                continue
            satir = f"SAHIPSIZ {s}: hesap {amt}, trend {trend} -> {fazla:+g} (~{abs(fazla)*fiyat:.0f} USDT, {yas:.0f} sa)"
            if kapat:
                ok_i, c_i = tt.emirleri_iptal(s) if e else (True, None)
                k = tt.ict_payini_kapat(s, amt, trend=trend)
                ok = bool(ok_i) and bool(k and k.get("ok"))
                tt.kaydet({"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "sembol": s, "tur": "DENETIM",
                           "sebep": "sahipsiz pozisyon", "basarili": ok, "pozisyon": amt,
                           "trend_payi": trend, "iptal": c_i, "kapanis": k})
                ozet["islem" if ok else "hata"] += 1
                satir += " -> KAPATILDI" if ok else f" -> HATA {k}"
            ozet["satirlar"].append(satir)
            continue

        if amt == 0 and e:
            eski = [x for x in e if (_yas_saat(_emir_zamani(x)) or 0) >= YAS_SAAT]
            if not eski:
                ozet["satirlar"].append(f"{s}: {len(e)} yeni emir (pozisyonsuz) - beklendi")
                continue
            satir = f"BAYAT EMIR {s}: {len(e)} emir, pozisyon yok"
            if kapat:
                ok, c = tt.emirleri_iptal(s)
                tt.kaydet({"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "sembol": s, "tur": "DENETIM",
                           "sebep": "bayat emir", "basarili": bool(ok), "sonuc": c})
                ozet["islem" if ok else "hata"] += 1
                satir += " -> IPTAL" if ok else f" -> HATA {c}"
            ozet["satirlar"].append(satir)
            continue

        if trend and abs(amt - trend) * fiyat >= tt.MIN_KAPAT_USDT:
            ozet["uyari"] += 1
            ozet["satirlar"].append(f"UYARI {s}: trend defteri {trend} ama hesap {amt} (dis mudahale?)")
        elif amt:
            ozet["satirlar"].append(f"{s}: trend payi {amt} - tamam")
    return ozet


def tg(metin):
    if not TG_TOKEN or not TG_CHAT:
        print(metin)
        return
    metin = metin.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    d = {"chat_id": TG_CHAT, "text": metin[:4090]}
    if TOPIC:
        d["message_thread_id"] = TOPIC
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data=d, timeout=20)
    except Exception as ex:
        print("TG hata:", ex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Testnet sahipsiz pozisyon/emir denetimi")
    ap.add_argument("--kapat", action="store_true", help="sahipsiz pozisyonlari kapat, bayat emirleri iptal et")
    a = ap.parse_args()
    oz = denetle(kapat=a.kapat)
    baslik = (f"TESTNET DENETIMI ({'temizlik' if a.kapat else 'sadece rapor'})\n"
              f"islem {oz['islem']} | uyari {oz['uyari']} | hata {oz['hata']}")
    print(baslik)
    for x in oz["satirlar"]:
        print(" ", x)
    # "anahtar yok" da onemli: yoksa kontrol YAPILMADIGI halde "temiz" raporu giderdi
    onemli = [x for x in oz["satirlar"] if x.startswith(("SAHIPSIZ", "BAYAT", "UYARI", "pozisyonlar",
                                                         "emir listesi", "anahtar yok"))]
    tg(baslik + ("\n\n" + "\n".join(onemli) if onemli else "\n\nSahipsiz pozisyon/bayat emir yok."))
