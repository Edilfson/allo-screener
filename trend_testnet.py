"""
TREND PORTFOYU -> BINANCE FUTURES TESTNET ESLEYICI
===================================================
trend_portfoy.py'nin 'b3' sepeti icin hesapladigi HEDEF AGIRLIKLARI
(state['portfoy']['b3']['w']) gunde bir kez testnet hesabinda 1x kaldiracla
gercek pozisyona cevirir. Kagit takip AYNEN devam eder; bu EK bir katmandir.

Kural:
  hedef_notional_i = TABAN * w_i          TABAN = env TREND_TESTNET_USDT (vars. 1000)
  hedef_miktar_i   = hedef_notional_i / fiyat_i   (LOT_SIZE stepSize + quantityPrecision)
  delta_i          = hedef_miktar_i - mevcut_miktar_i   (positionRisk.positionAmt)
  |delta_i * fiyat_i| < 25 USDT ise EMIR YOK (min notional + gurultu filtresi)
  emir tipi MARKET; azaltma yonunde reduceOnly=true; kaldirac 1 (marjin tipine dokunulmaz)

ICT SCREENER ILE ORTAK HESAP:
  - Bu modul SADECE MARKET pozisyon ayari yapar; hicbir acik emri (limit/stop/TP)
    iptal etmez (allOpenOrders/algoOpenOrders'a DOKUNMAZ).
  - Ayni sembolde son 24 saatte basarili bir ICT giris emri kaydi varsa
    (testnet_orders.json) o sembol ATLANIR ve loglanir; yoksa ICT'nin stop/TP'si
    bizim MARKET emrimizle karisir.

KURU MOD (--kuru): API'ye HICBIR yazma yapilmaz, sadece hesap yazdirilir ve
testnet_orders.json'a da yazilmaz. TESTNET_API_KEY bossa otomatik kuru moda dusulur
(fiyat/exchangeInfo herkese acik uclardan okunur, pozisyon 0 varsayilir).

UYARI: Testnet likiditesi gercek piyasadan farklidir. Yatirim tavsiyesi degildir.
Secret: TESTNET_API_KEY, TESTNET_API_SECRET
"""

import argparse
import json
import os
import time

import requests

import testnet_trader as tt

BASE = tt.BASE
KEY = os.environ.get("TESTNET_API_KEY", "")
SECRET = os.environ.get("TESTNET_API_SECRET", "")

TABAN = float(os.environ.get("TREND_TESTNET_USDT", "1000"))   # eslenecek toplam sermaye
KALDIRAC = 1                       # trend portfoyu kaldiracsizdir
MIN_NOTIONAL = 25.0                # bu tutarin altindaki delta islenmez
ICT_PENCERE = 24 * 3600            # ICT emri "taze" sayilan sure (saniye)
POZ_DOSYA = "positions.json"       # screener pozisyonlari (ayni repo checkout'u)


def fiyat(sembol):
    """GET /fapi/v1/ticker/price - imza gerektirmez (kuru modda da calisir)."""
    try:
        r = requests.get(f"{BASE}/fapi/v1/ticker/price",
                         params={"symbol": sembol}, timeout=20)
        return float(r.json()["price"])
    except Exception:
        return None


def mevcut_miktar(sembol):
    """GET /fapi/v2/positionRisk -> positionAmt toplami.
    Pozisyon yoksa 0.0, okunamazsa None (hedge modda iki bacak toplanir)."""
    ok, c = tt._imzali("/fapi/v2/positionRisk", {"symbol": sembol})
    if not ok:
        return None
    toplam = 0.0
    for x in (c if isinstance(c, list) else []):
        if x.get("symbol") == sembol:
            try:
                toplam += float(x.get("positionAmt", 0) or 0)
            except Exception:
                pass
    return toplam


def ict_emri_var_mi(sembol, pencere=ICT_PENCERE):
    """testnet_orders.json'da son `pencere` saniyede o sembol icin BASARILI bir
    ICT giris emri kaydi var mi? (tur=TREND bizim kayitlarimiz, IPTAL kayitlari
    giris emri degil -> ikisi de sayilmaz; ICT giris kayitlarinda 'strateji' alani var)"""
    try:
        L = json.load(open(tt.STATE)) if os.path.exists(tt.STATE) else []
    except Exception:
        return False
    simdi = time.time()
    for r in reversed(L[-500:]):
        if not isinstance(r, dict) or r.get("sembol") != sembol:
            continue
        if r.get("tur") == "TREND" or not r.get("basarili") or not r.get("strateji"):
            continue
        try:
            t = time.mktime(time.strptime(str(r.get("zaman", ""))[:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            continue
        if 0 <= simdi - t <= pencere:
            return True
    return False


def screener_pozisyonu_var(sembol):
    """positions.json'da o sembolde testnete GONDERILMIS acik/bekleyen ICT pozisyonu var mi?
    (24 saatlik emir penceresinden bagimsiz: ICT pozisyonu gunlerce acik kalabilir)"""
    try:
        P = json.load(open(POZ_DOSYA)) if os.path.exists(POZ_DOSYA) else []
    except Exception:
        return False
    return any(isinstance(x, dict) and x.get("symbol") == sembol
               and x.get("status") in ("open", "pending") and x.get("sinyal_gonderildi")
               for x in P)


def defter_kur():
    """KENDI pozisyon defterimizi testnet_orders.json'daki BASARILI TREND kayitlarindan kurar.
    Her kaydin sonuc.hedef_miktar'i = o emirden sonra trend'in tuttugu miktar (MARKET emir tam dolar)."""
    try:
        L = json.load(open(tt.STATE)) if os.path.exists(tt.STATE) else []
    except Exception:
        return {}
    d = {}
    for r in L:
        if isinstance(r, dict) and r.get("tur") == "TREND" and r.get("basarili") is True:
            s = r.get("sonuc") or {}
            if "hedef_miktar" in s:
                d[r["sembol"]] = float(s["hedef_miktar"])
    return d


def _kayit(kuru, sembol, w, hedef_notional, mevcut, delta, basarili, sonuc):
    """Tek satirlik kayit. basarili: True/False = emir gonderildi/hata,
    None = emir gonderilmedi (atlandi). Kuru modda dosyaya YAZILMAZ."""
    k = {"zaman": time.strftime("%Y-%m-%dT%H:%M:%S"), "tur": "TREND", "sembol": sembol,
         "hedef_w": round(float(w), 6), "hedef_notional": round(float(hedef_notional), 2),
         "mevcut_miktar": mevcut, "delta": delta, "basarili": basarili, "sonuc": sonuc}
    if not kuru:
        tt.kaydet(k)
    return k


def esle(agirliklar, kuru=False, defter=None):
    """Hedef agirliklari testnet pozisyonlariyla eslestirir.
    agirliklar: {"BTCUSDT": 0.14, ...}   Return: ozet dict."""
    anahtar_var = bool(KEY and SECRET)
    kuru = bool(kuru) or not anahtar_var
    # DEFTER: trend'in KENDI tuttugu miktar (sembol -> adet). Delta hesap toplamina degil buna gore
    # hesaplanir; boylece ayni hesaptaki ICT pozisyonlari (veya sahipsiz kalintilar) degistirilmez.
    # (2026-09-13: ilk canli kosuda hesaptaki 4.19 BNB ICT kalintisi trend'inmis gibi 0.2'ye indirildi)
    if defter is None:
        defter = defter_kur()
    ozet = {"kuru": kuru, "taban": TABAN, "emir": 0, "atlanan": 0, "hata": 0,
            "satirlar": [], "kayitlar": [], "defter": defter}
    if not agirliklar:
        ozet["satirlar"].append("agirlik yok, yapilacak islem yok")
        print("  [trend-testnet] agirlik yok")
        return ozet

    for sembol in sorted(agirliklar):
        w = float(agirliklar.get(sembol) or 0)
        hedef_notional = TABAN * w
        p = fiyat(sembol)
        if not p:
            ozet["hata"] += 1
            ozet["kayitlar"].append(_kayit(kuru, sembol, w, hedef_notional, None, None,
                                           False, {"hata": "fiyat okunamadi"}))
            ozet["satirlar"].append(f"{sembol}: fiyat okunamadi")
            continue

        _tick, adim, qp, _pp, _maxq = tt.sembol_bilgi(sembol)
        hedef_miktar = tt._yuvarla(hedef_notional / p, adim, qp)

        mevcut = mevcut_miktar(sembol) if anahtar_var else 0.0
        if mevcut is None:
            ozet["hata"] += 1
            ozet["kayitlar"].append(_kayit(kuru, sembol, w, hedef_notional, None, None,
                                           False, {"hata": "pozisyon okunamadi"}))
            ozet["satirlar"].append(f"{sembol}: pozisyon okunamadi")
            continue

        # ICT screener ayni sembolde aktifse -> karismamak icin atla
        neden = ("son 24s ICT emri var" if ict_emri_var_mi(sembol)
                 else "screener acik pozisyonu var" if screener_pozisyonu_var(sembol) else None)
        if neden:
            ozet["atlanan"] += 1
            ozet["kayitlar"].append(_kayit(kuru, sembol, w, hedef_notional, mevcut, None,
                                           None, {"atlandi": neden}))
            ozet["satirlar"].append(f"{sembol}: {neden}, atlandi")
            print(f"  [trend-testnet] {sembol}: {neden}, atlandi")
            continue

        bizim = float(defter.get(sembol, 0.0) or 0.0)
        delta = round(hedef_miktar - bizim, 12)
        miktar = tt._yuvarla(abs(delta), adim, qp)
        if abs(delta) * p < MIN_NOTIONAL or miktar <= 0:
            ozet["atlanan"] += 1
            ozet["kayitlar"].append(_kayit(kuru, sembol, w, hedef_notional, mevcut, delta,
                                           None, {"atlandi": "min notional",
                                                  "delta_usdt": round(abs(delta) * p, 2)}))
            ozet["satirlar"].append(
                f"{sembol}: hedef {hedef_miktar} / bizim {bizim} (hesap {mevcut}) -> "
                f"delta {delta} (~{abs(delta)*p:.1f} USDT) < {MIN_NOTIONAL:.0f}, atlandi")
            continue

        yon = "BUY" if delta > 0 else "SELL"
        # azaltma = mutlak pozisyon kuculuyor ve yon degismiyor -> reduceOnly guvenli
        # reduceOnly SADECE: kendi payimiz kuculuyor, yon degismiyor VE hesapta en az o kadar
        # ayni yonlu pozisyon var (yoksa borsa reddeder; o durumda duz MARKET ile ters islem)
        azaltma = (bizim != 0 and abs(hedef_miktar) < abs(bizim) and hedef_miktar * bizim >= 0
                   and mevcut * bizim > 0 and abs(mevcut) >= miktar)
        satir = (f"{sembol}: w=%{w*100:.1f} hedef {hedef_miktar} ({hedef_notional:.0f} USDT) "
                 f"| bizim {bizim} (hesap {mevcut}) | {yon} {miktar}" + (" reduceOnly" if azaltma else ""))

        if kuru:
            ozet["emir"] += 1
            ozet["satirlar"].append("[kuru] " + satir)
            print("  [trend-testnet][kuru] " + satir)
            continue

        tt._imzali("/fapi/v1/leverage", {"symbol": sembol, "leverage": KALDIRAC}, "POST")
        params = {"symbol": sembol, "side": yon, "type": "MARKET", "quantity": miktar}
        if azaltma:
            params["reduceOnly"] = "true"
        ok, c = tt._imzali("/fapi/v1/order", params, "POST")
        if ok:
            ozet["emir"] += 1
            defter[sembol] = round(bizim + (miktar if yon == "BUY" else -miktar), 12)
        else:
            ozet["hata"] += 1
        sonuc = {"yon": yon, "miktar": miktar, "fiyat": p, "reduceOnly": azaltma,
                 "hedef_miktar": hedef_miktar, "bizim_miktar": bizim, "yanit": c}
        ozet["kayitlar"].append(_kayit(kuru, sembol, w, hedef_notional, mevcut, delta,
                                       bool(ok), sonuc))
        ozet["satirlar"].append(satir + (" -> OK" if ok else f" -> HATA {c}"))
        print(f"  [trend-testnet] {satir} -> {'OK' if ok else 'HATA ' + str(c)}")

    print(f"  [trend-testnet] ozet: {ozet['emir']} emir, {ozet['atlanan']} atlandi, "
          f"{ozet['hata']} hata{' (kuru)' if kuru else ''}")
    return ozet


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trend portfoyunu (b3) testnete esle")
    ap.add_argument("--kuru", action="store_true", help="API'ye yazma, sadece hesapla")
    args = ap.parse_args()

    st = json.load(open("trend_state.json")) if os.path.exists("trend_state.json") else {}
    w = ((st.get("portfoy") or {}).get("b3") or {}).get("w") or {}
    if not w:
        print("trend_state.json icinde portfoy.b3.w yok - once trend_bot.py calistirilmali")
    print(f"TABAN {TABAN:.0f} USDT | kaldirac {KALDIRAC}x | min notional {MIN_NOTIONAL:.0f} USDT")
    ozet = esle(w, kuru=args.kuru)
    for s in ozet["satirlar"]:
        print(" ", s)
