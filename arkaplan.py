"""ARKA PLAN STRATEJILERI - KAGIT, ILERIYE DONUK TAKIP (2026-09-13)
====================================================================
KULLANICI KARARI: demo hesapta SADECE trend portfoyu (trend_testnet.py) islem yapar. Lab'da test edilip ELENEN
stratejiler burada gercek PUBLIC piyasa verisiyle, baslangic gununden itibaren ileriye donuk KAGIT uzerinde izlenir.
Bilgi amaclidir: HICBIR emir gondermez, anahtar kullanmaz. Amac gercek out-of-sample kanit uretmek.
Ilk calismada gecmis simulasyon YOK; gecmis mumlar sadece gosterge isinmasi icindir.

STRATEJILER (her biri bagimsiz kagit hesap; parametre kaynaklari):
 a) ema_geri_1d       araclar/lab/ema_geri.py strateji_sinyal + sinyal_yap: tf=1d, EMA 20/50, Y=0.15, P=7, R=20,
                      cikis=tp3, filtre=f0, max_days 1d=60, likidite 30g ort USDT hacmi >= 5M (MIN_VOL). Islem bazli R.
 b) kirilim_1d        araclar/lab/kirilim_gunluk.py sinyaller: d_N40_M20_k3.0_long_tp5R (sonuc_kirilim_gunluk.json
                      en_iyi_3[0]): kapanis > onceki 40 gunun en yuksek kapanisi, stop ref-3*ATR14, TP 5R, cikis ilk
                      20 gunluk dusuk kapanis alti (i+2'den itibaren), MAX_DAYS=365, 30g hacim >= 5M. Islem bazli R.
 c) momentum_haftalik araclar/lab/kesitsel.py build_W H1_K14_U30_L20_REJVOL (sonuc_kesitsel.json en_iyi_3[0]):
                      mom = C[t-7]/C[t-21]-1 (K=14, skip=7), evren 30g ort hacme gore ilk 30 (gecmis >= 240 gun:
                      gunluk_panel(min_gun=120) + hist>=120), en iyi 20 long, agirlik ~ 1/vol30 (toplam 1),
                      W[t] = baz * BTC(close>MA50)[t] HER GUN; baz haftalik. Maliyet |dW|*0.0007 (sim_portfoy).
 d) tersine_haftalik  araclar/lab/tersine.py build_W LS_L3_N10_U30_rb7: x = C[t]/C[t-3]-1, evren ilk 30 (gecmis >= 150
                      gun: panel(min_gun=120) + tam 30g hacim penceresi), en kotu 10 +0.05, en iyi 10 -0.05;
                      maliyet 0.0007 + short notional * 0.0003/gun (tersine.sim).
 e) funding_carry     araclar/lab/funding.py w_esik(E=0.10, mod=sabit_slot) (H1b_esik10): CEKIRDEK evren;
                      f7 = 7g ort gunluk funding * 365; > E ac, < E/2 kapa (histerezis); notional/sermaye = 0.5/K;
                      getiri/notional x[d] = funding(d) + (spot acilis getiri - perp acilis getiri); maliyet 0.19% |dW|.
 f) OKUYUCULAR (hesaplama yok): ICT = positions.json baslangic tarihinden sonra ACILAN dolmus+kapanmis islemler
    (screener.py net_r mantigi), trend = trend_state.json portfoy.b3 / yuruyen3 / yuruyen5_tavan, BTC al-tut.

SIMULASYON (lab ile birebir, gun gun artimli):
 - Islem bazli: harness.sim_islem kurallari. Sinyal t kapanisi, giris t+1 acilisi * (1+kayma); acilis stopun altindaysa
   islem yok (kilit yok). Her gun sira: zaman asimi (onceki kapanis) -> stop (-1R, giris mumunda da) -> TP (giris mumu
   haric) -> kanal cikisi (kapanis). Net R = R - (taker giris 0.05% + taker cikis 0.05% + kayma 0.05%) / stop%.
   Sembol kilidi: acik/bekleyen pozisyon veya ayni gun cikis varsa o sembolde yeni sinyal alinmaz.
 - Portfoy: harness.sim_portfoy adimi. W[t] t kapanisinda karar, t+1 acilisinda uygulanir, getiri O[t+1]->O[t+2].
   Sadece kapanmis mumla calismak icin t adimi D=t+2 gunu islenirken gerceklesir (equity O[D]'ye kadar isaretli).
 - Ayni gun tekrar kosarsa strateji basina son_gun sayesinde hicbir sey cift sayilmaz; kacan gunler sirayla islenir.
 - Bir stratejinin hatasi digerlerini durdurmaz (strateji kopyasi uzerinde calisilir, hata olursa geri alinir).

BILINCLI SAPMALAR (lab tanimina gore):
 1) Evren: bugunku vadeli USDT perpetual listesi (altin/doviz/stable haric) icinde 30g ort quote hacmine gore ilk 50
    (lab: tum ~350 coin + >=5M likidite). Momentum/tersine ilk-30'u bu listeden (gun t'deki hacimle) secer.
 2) Gosterge isinmasi son 400 gunluk mumla (lab: listelemeden beri). EMA50 farki ~1e-6 duzeyi.
 3) ema_geri min 150 bar / kirilim min 200 bar kosulu nedensel uygulanir (lab toplam seri uzunluguna bakiyordu).
 4) Haftalik dengeleme PAZARTESI kapanisinda (lab: baslangictan itibaren 7 gunde bir; sadece faz farki).
    Ilk pazartesiye kadar momentum/tersine nakitte.
 5) funding_carry evreni CEKIRDEK-13 (lab VALID adayi H1b_esik10__genis: CEKIRDEK U aylik top-20 idi). Histerezis
    baslangic durumu kapali. Funding verisi son <=30 gun indirilir.
 6) Islem bazli kanal/zaman cikisi lab'daki gibi o gunun (onceki gunun) KAPANIS fiyatindan; t+1 acilisi degil.
UYARI: yatirim tavsiyesi degildir; kagit takiptir.

Kullanim:
  python arkaplan.py            # isle + state yaz + (pazartesi) Telegram
  python arkaplan.py --kuru     # gercek veriyle bugunku sinyal/agirliklari yazdir; state YAZMA, Telegram GONDERME
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TOPIC_SUMMARY, ARKAPLAN_HER_GUN=1 (her kosuda rapor)
"""
import argparse
import copy
import html
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import requests

SPOT = "https://data-api.binance.vision"
FAPI = "https://fapi.binance.com"
STATE_FILE = "arkaplan_state.json"
TREND_STATE = "trend_state.json"
ICT_FILE = "positions.json"
GUN_MS = 86400_000

# ---------------------------------------------------------------- veri / evren
EVREN_N = 50
ADAY_24S = 80              # 30g hacim siralamasi icin mum indirilen aday sayisi (24s hacme gore)
KLINE_LIMIT = 400
MAKS_ISTEK = 240
MAKS_SURE = 280            # saniye
ISCI = 8
# screener.py get_usdt_symbols haric listesi (altin/emtia/doviz)
HARIC = ("PAXG", "XAUT", "XAU", "TGOLD", "KAU", "AUX", "XAG", "WPAXG",
         "EURI", "AEUR", "EUR", "TRY", "BRL", "ARS", "ZAR", "JPY")

# ---------------------------------------------------------------- islem bazli (harness.MALIYET, piyasa girisi)
FEE_TAKER = 0.0005
SLIP = 0.0005
MALIYET_ISLEM = FEE_TAKER + FEE_TAKER + SLIP
MIN_VOL = 5e6              # ema_geri.MIN_VOL, kirilim_gunluk.MIN_VOL
EG = dict(ef=20, es=50, Y=0.15, P=7, R=20, tpR=3.0, max_gun=60, min_bar=150)          # ema_geri.py 1d
KG = dict(N=40, M=20, k=3.0, tpR=5.0, max_gun=365, min_bar=200)                      # kirilim_gunluk.py
# ---------------------------------------------------------------- portfoy
MOM = dict(K=14, skip=7, U=30, L=20, pencere=30, min_bar=240, maliyet=0.0007)        # kesitsel.py
TER = dict(L=3, N=10, U=30, pencere=30, min_bar=150, maliyet=0.0007, funding=0.0003) # tersine.py
CEKIRDEK = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT",
            "LINKUSDT", "BCHUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"]                  # funding.py CEKIRDEK
FND = dict(E=0.10, notional=0.5, maliyet=0.0010 + 0.0005 + 2 * 0.0002, uyumsuz=0.30, gun=30)  # funding.py
ICT_MALIYET = 0.0002 + 0.0005 + 0.0005     # screener.py FEE_MAKER + FEE_TAKER + SLIPPAGE (net_r)

KAPANAN_MAKS = 200
SERI_MAKS = 2000
_uyu = time.sleep


# ================================================================ yardimcilar
def gun_str(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")


def gun_ms(g):
    return int(datetime.strptime(g, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def gun_ekle(g, k):
    return gun_str(gun_ms(g) + k * GUN_MS)


def gun_fark(a, b):
    return (gun_ms(b) - gun_ms(a)) // GUN_MS


def pazartesi_mi(g):
    return datetime.strptime(g, "%Y-%m-%d").weekday() == 0


def _f(x):
    return x is not None and np.isfinite(x)


class Istemci:
    """Anahtarsiz GET; istek butcesi + sure siniri + 429/418'de bekle-tekrar."""

    def __init__(self, maks=MAKS_ISTEK, sure=MAKS_SURE):
        self.say = 0
        self.maks = maks
        self.sure = sure
        self.t0 = time.time()
        self._kilit = threading.Lock()

    def get(self, url, params=None):
        for _ in range(4):
            with self._kilit:
                if self.say >= self.maks:
                    raise RuntimeError(f"istek butcesi asildi ({self.maks})")
                if time.time() - self.t0 > self.sure:
                    raise RuntimeError("sure siniri asildi")
                self.say += 1
            r = requests.get(url, params=params, timeout=30)
            if r.status_code in (429, 418):
                bas = getattr(r, "headers", None) or {}
                try:
                    bekle = float(bas.get("Retry-After", 5))
                except (TypeError, ValueError):
                    bekle = 5.0
                _uyu(min(max(bekle, 1.0), 30.0))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {url}")
            return r.json()
        raise RuntimeError(f"429 tekrarlari tukendi: {url}")


class Seri:
    """Gunluk mumlar (SADECE kapanmis). gun = acilis tarihi YYYY-MM-DD."""

    def __init__(self, satirlar, simdi_ms=None, perp=False):
        rows = [x for x in satirlar if simdi_ms is None or int(x[6]) < simdi_ms]
        if perp:   # funding.py: olu perp piyasasi (hacim 0 veya h==l donmus mum) gecersiz
            rows = [x for x in rows if float(x[5]) > 0 and float(x[2]) > float(x[3])]
        self.ot = np.array([int(x[0]) for x in rows], dtype=np.int64)
        self.o, self.h, self.l, self.c, self.qv = (np.array([float(x[k]) for x in rows], dtype=float)
                                                   for k in (1, 2, 3, 4, 7))
        self.gun = [gun_str(t) for t in self.ot]
        self.ix = {g: i for i, g in enumerate(self.gun)}

    @classmethod
    def diziden(cls, ot, o, h, l, c, qv):
        rows = [[int(ot[i]), o[i], h[i], l[i], c[i], 1.0, int(ot[i]) + GUN_MS - 1, qv[i]] for i in range(len(ot))]
        return cls(rows)

    def acilis(self, g):
        i = self.ix.get(g)
        return float(self.o[i]) if i is not None else float("nan")


class Veri:
    def __init__(self, seri, evren, takvim, fund=None, perp=None, hatalar=None):
        self.seri = seri            # sym -> Seri (spot)
        self.evren = evren          # bugunku ilk-50
        self.takvim = takvim        # BTC kapanmis gunleri
        self.fund = fund            # sym -> {gun: gunluk funding toplami} | None (alinamadi)
        self.perp = perp or {}      # sym -> Seri (perp)
        self.hatalar = hatalar or {}


# ================================================================ gostergeler (ema_geri.py / kirilim_gunluk.py port)
def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty(len(x))
    m = x[0]
    for i, v in enumerate(x):
        m = a * v + (1 - a) * m
        out[i] = m
    return out


def roll_mean(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def atr14(h, l, c):
    pc = np.concatenate([[c[0]], c[:-1]])
    return roll_mean(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), 14)


def _roll_max(x, N):
    """out[i] = max(x[i-N .. i-1])  (kirilim_gunluk._roll_max)"""
    n = len(x)
    out = np.full(n, np.nan)
    if n > N:
        out[N:] = np.lib.stride_tricks.sliding_window_view(x, N)[:n - N].max(axis=1)
    return out


def ema_geri_sinyaller(h, l, c, qv, bas=0):
    """ema_geri.strateji_sinyal(tf=1d, ema=(20,50), Y=0.15, P=7, cikis=tp3, filtre=f0) ile ayni kural.
    Fark: veri sonundaki 2 bar KESILMEZ (canli: son kapanmis gun sinyal verebilir). bas: aday indeks alt siniri."""
    ef, es, Y, P, R = EG["ef"], EG["es"], EG["Y"], EG["P"], EG["R"]
    n = len(c)
    if n < max(R + 1, 2):
        return []
    e1, e2 = ema(c, ef), ema(c, es)
    a = atr14(h, l, c)
    v30 = roll_mean(qv, 30)
    with np.errstate(invalid="ignore"):
        liq = np.isfinite(v30) & (v30 >= MIN_VOL)
        i = np.arange(n)
        ok = i >= max(R, 10)
        e2l = np.full(n, np.inf); e2l[10:] = e2[:-10]
        cl = np.full(n, np.inf); cl[R:] = c[:-R]
        mx = np.full(n, -np.inf); mx[R:] = np.lib.stride_tricks.sliding_window_view(c, R + 1).max(axis=1)
        tr = ok & (e1 > e2) & (e2 > e2l) & (mx >= cl * (1 + Y)) & liq
        warm = max(2 * es, R + P + 15, 60)
        tr[:warm] = False
        tr[:bas] = False
        trig = np.zeros(n, bool)
        trig[1:] = (c[1:] > e1[1:]) & (c[1:] > h[:-1])
        aday = np.flatnonzero(tr & trig & np.isfinite(a) & (a > 0))
        lim = e2 - 0.5 * a
        touch = (l <= e1) & (c >= lim)
        bozuk = c < lim
    out = []
    for i in aday:
        i = int(i)
        w0 = i - P
        if bozuk[w0:i].any():
            continue
        pk = (i - R) + int(np.argmax(h[i - R:i]))
        if i - pk < 3:
            continue
        k0 = max(w0, pk + 1)
        if not touch[k0:i].any():
            continue
        stop = float(l[w0:i].min() - 0.5 * a[i])
        sp = (c[i] - stop) / c[i]
        if sp < 0.005 or sp > 0.15:
            continue
        out.append(dict(i=i, stop=stop, tp=float(c[i] + EG["tpR"] * (c[i] - stop))))
    return out


def kirilim_sinyaller(h, l, c, qv, bas=0):
    """kirilim_gunluk.sinyaller(N=40, M=20, k=3.0, yon=long, tp=5, cikis=kanal, giris=piyasa) ile ayni tetik/stop/TP.
    Fark: son 2 bar kesilmez. Kanal cikisi kirilim_kanal_cikis ile gun gun kontrol edilir."""
    N, M, k = KG["N"], KG["M"], KG["k"]
    n = len(c)
    if n <= N:
        return []
    pc = np.concatenate([[c[0]], c[:-1]])
    atr = roll_mean(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), 14)
    vol30 = roll_mean(qv, 30)
    hi = _roll_max(c, N)
    with np.errstate(invalid="ignore"):
        trig = (c > hi) & np.isfinite(atr) & (atr > 0) & np.isfinite(vol30) & (vol30 >= MIN_VOL)
    warm = max(N, M, 30, 14) + 1
    trig[:max(warm, bas)] = False
    out = []
    for i in np.flatnonzero(trig):
        i = int(i)
        risk = k * atr[i]
        stop = c[i] - risk
        if stop <= 0:
            continue
        out.append(dict(i=i, stop=float(stop), tp=float(c[i] + KG["tpR"] * risk)))
    return out


def kirilim_kanal_cikis(c, j, M=KG["M"]):
    """kirilim_gunluk exit_cond: c[j] < min(c[j-M .. j-1])"""
    return j >= M and c[j] < c[j - M:j].min()


def _son_sinyal(fn, min_bar):
    def f(s, j):
        n = j + 1
        if n < min_bar:
            return None
        sg = fn(s.h[:n], s.l[:n], s.c[:n], s.qv[:n], bas=j)     # dilim: ileriye bakma fiziksel olarak imkansiz
        return sg[-1] if sg and sg[-1]["i"] == j else None
    return f


EMA_CFG = dict(sinyal=_son_sinyal(ema_geri_sinyaller, EG["min_bar"]), max_gun=EG["max_gun"], kanal=False)
KIRILIM_CFG = dict(sinyal=_son_sinyal(kirilim_sinyaller, KG["min_bar"]), max_gun=KG["max_gun"], kanal=True)


# ================================================================ islem bazli motor (harness.sim_islem, artimli)
def islem_gun(st, V, D, cfg):
    olay = []
    kilitli = set()
    kalan = []
    for p in st["pozisyonlar"]:
        sym = p["sym"]
        s = V.seri.get(sym)
        j = s.ix.get(D) if s is not None else None
        kilitli.add(sym)
        if j is None:                      # o gun veri yok: bekle
            kalan.append(p)
            continue
        if p["durum"] == "bekliyor":       # giris: sinyal ertesi acilis + kayma
            e = float(s.o[j]) * (1 + SLIP)
            if e <= p["stop"]:
                st["iptal"] += 1
                olay.append(f"{sym} giris iptal (acilis {e:.6g} <= stop {p['stop']:.6g})")
                kilitli.discard(sym)       # sim_islem: kilit konmaz
                continue
            p.update(durum="acik", giris=e, risk=e - p["stop"], sp=(e - p["stop"]) / e, giris_gun=D)
            olay.append(f"{sym} GIRIS {e:.6g} stop {p['stop']:.6g} tp {p['tp']:.6g}")
        e, risk = p["giris"], p["risk"]
        res = r = None
        cikis_gun = D
        if gun_fark(p["sinyal_gun"], D) > 1 + cfg["max_gun"] and j >= 1:
            res, r, cikis_gun = "timeout", (float(s.c[j - 1]) - e) / risk, s.gun[j - 1]
        elif s.l[j] <= p["stop"]:
            res, r = "stopped", -1.0
        elif p.get("tp") is not None and D != p["giris_gun"] and s.h[j] >= p["tp"]:
            res, r = "target_done", (p["tp"] - e) / risk
        elif cfg["kanal"] and D != p["giris_gun"] and kirilim_kanal_cikis(s.c, j):
            res, r = "cikis", (float(s.c[j]) - e) / risk
        if res is None:
            p["son_r"] = round((float(s.c[j]) - e) / risk, 4)
            kalan.append(p)
            continue
        net = r - MALIYET_ISLEM / p["sp"]
        st["kapanan"].append(dict(sym=sym, sinyal_gun=p["sinyal_gun"], giris_gun=p["giris_gun"], cikis_gun=cikis_gun,
                                  giris=e, stop=p["stop"], tp=p.get("tp"), res=res, r=round(r, 6), net=round(net, 6)))
        st["kapanan"] = st["kapanan"][-KAPANAN_MAKS:]
        st["islem_sayisi"] += 1
        st["toplam_net_r"] += net
        st["kazanan"] += int(net > 0)
        olay.append(f"{sym} CIKIS {res} R={r:+.2f} net={net:+.2f}")
    for sym in V.evren:
        if sym in kilitli:
            continue
        s = V.seri.get(sym)
        j = s.ix.get(D) if s is not None else None
        if j is None:
            continue
        sg = cfg["sinyal"](s, j)
        if sg:
            c = float(s.c[j])
            kalan.append(dict(sym=sym, sinyal_gun=D, durum="bekliyor", stop=sg["stop"], tp=sg["tp"], ref=c))
            olay.append(f"{sym} SINYAL kapanis {c:.6g} stop {sg['stop']:.6g} (-%{(c - sg['stop']) / c * 100:.1f}) "
                        f"tp {sg['tp']:.6g} -> giris {gun_ekle(D, 1)} acilis")
    st["pozisyonlar"] = kalan
    st["seri"].append([D, round(st["toplam_net_r"], 4), sum(1 for p in kalan if p["durum"] == "acik")])
    st["seri"] = st["seri"][-SERI_MAKS:]
    return olay


# ================================================================ portfoy motoru (harness.sim_portfoy adimi)
def portfoy_adim(st, D, getiri_fn, maliyet, short_funding=0.0):
    """t = D-2: W[t] O[t+1]'de uygulanir, getiri O[t+1] -> O[t+2] = O[D]. getiri_fn(s, t) -> (O[t+1] var mi, r)."""
    t = gun_ekle(D, -2)
    W = st["hedef"].get(t)
    if W is None:
        return None
    w, r = {}, {}
    for s, v in W.items():
        gecerli, rr = getiri_fn(s, t)
        if gecerli and v != 0:
            w[s] = v
            r[s] = rr
    wp = st["w_prev"]
    semb = set(w) | set(wp)
    x = sum(w[s] * r[s] for s in w) - sum(abs(w.get(s, 0.0) - wp.get(s, 0.0)) for s in semb) * maliyet
    S = sum(-v for v in w.values() if v < 0)
    g = x - short_funding * S            # tersine.sim: funding surukleme hesabina girmez
    st["equity"] *= 1 + g
    st["w_prev"] = {s: w[s] * (1 + r[s]) / (1 + x) for s in w} if x > -1 else dict(w)
    st["zirve"] = max(st["zirve"], st["equity"])
    st["dusus"] = round(1 - st["equity"] / st["zirve"], 6)
    return g


def hedef_kaydet(st, D, W):
    onceki = st["hedef"].get(gun_ekle(D, -1), {})
    st["islem_sayisi"] += sum(1 for s in set(W) | set(onceki) if abs(W.get(s, 0.0) - onceki.get(s, 0.0)) > 1e-12)
    st["hedef"][D] = W
    sinir = gun_ekle(D, -1)
    st["hedef"] = {g: v for g, v in st["hedef"].items() if g >= sinir}


def spot_getiri(V):
    def f(s, t):
        se = V.seri.get(s)
        if se is None:
            return False, 0.0
        o1, o2 = se.acilis(gun_ekle(t, 1)), se.acilis(gun_ekle(t, 2))
        if not np.isfinite(o1):
            return False, 0.0
        r = o2 / o1 - 1 if np.isfinite(o2) else 0.0
        return True, float(np.clip(r, -0.95, 5.0))
    return f


def btc_rejim(V, D):
    s = V.seri.get("BTCUSDT")
    j = s.ix.get(D) if s is not None else None
    if j is None or j < 49:
        return 0
    return int(s.c[j] > s.c[j - 49:j + 1].mean())


def _evren_adv(V, D, min_bar, pencere):
    out = []
    for s in V.evren:
        se = V.seri.get(s)
        j = se.ix.get(D) if se is not None else None
        if j is None or j + 1 < min_bar:
            continue
        adv = float(se.qv[j - pencere + 1:j + 1].mean())
        if np.isfinite(adv) and adv > 0:
            out.append((s, j, adv))
    out.sort(key=lambda x: -x[2])
    return out


def momentum_baz(V, D):
    """kesitsel.build_W(K=14, U=30, L=20, mod=long, voltgt=True) dengeleme gunu agirliklari (rejimsiz)."""
    ev = _evren_adv(V, D, MOM["min_bar"], MOM["pencere"])[:MOM["U"]]
    K, sk = MOM["K"], MOM["skip"]
    mom = []
    for s, j, _ in ev:
        c = V.seri[s].c
        m = c[j - sk] / c[j - sk - K] - 1.0
        if np.isfinite(m):
            mom.append((s, j, m))
    if len(mom) < MOM["L"]:
        return {}
    sec = sorted(mom, key=lambda x: -x[2])[:MOM["L"]]
    iv = []
    for s, j, _ in sec:
        c = V.seri[s].c
        rr = c[j - MOM["pencere"] + 1:j + 1] / c[j - MOM["pencere"]:j] - 1.0
        vol = np.sqrt(max(float(np.mean(rr ** 2) - np.mean(rr) ** 2), 0.0))
        v = 1.0 / max(vol, 1e-4)
        iv.append(v if np.isfinite(v) else 0.0)
    iv = np.array(iv)
    if iv.sum() <= 0:
        iv = np.ones(len(sec))
    return {s: float(v) for (s, _, _), v in zip(sec, iv / iv.sum())}


def tersine_baz(V, D):
    """tersine.build_W(L=3, N=10, U=30, mod=ls, sig=ret) dengeleme gunu agirliklari."""
    ad = _evren_adv(V, D, TER["min_bar"], TER["pencere"])
    if len(ad) < TER["U"]:
        return {}
    ev = ad[:TER["U"]]
    x = []
    for s, j, _ in ev:
        c = V.seri[s].c
        v = c[j] / c[j - TER["L"]] - 1.0
        if np.isfinite(v):
            x.append((s, v))
    N = TER["N"]
    if len(x) < max(2 * N, int(0.8 * TER["U"])):
        return {}
    srt = sorted(x, key=lambda z: z[1])          # kararli siralama (argsort kind=stable)
    W = {s: 0.5 / N for s, _ in srt[:N]}
    for s, _ in srt[-N:]:
        W[s] = -0.5 / N
    return W


def momentum_gun(st, V, D):
    portfoy_adim(st, D, spot_getiri(V), MOM["maliyet"])
    rej = btc_rejim(V, D)
    olay = []
    if pazartesi_mi(D):
        st["baz"] = momentum_baz(V, D)
        st["son_dengeleme"] = D
        olay.append(f"PAZARTESI dengeleme: {len(st['baz'])} coin")
    W = dict(st["baz"]) if rej == 1 else {}
    st["rejim"] = rej
    hedef_kaydet(st, D, W)
    st["seri"].append([D, round(st["equity"], 6)])
    st["seri"] = st["seri"][-SERI_MAKS:]
    olay.append(f"BTC>MA50={'evet' if rej else 'hayir'} brut %{sum(W.values()) * 100:.0f}, {len(W)} coin")
    return olay


def tersine_gun(st, V, D):
    portfoy_adim(st, D, spot_getiri(V), TER["maliyet"], short_funding=TER["funding"])
    olay = []
    if pazartesi_mi(D):
        st["baz"] = tersine_baz(V, D)
        st["son_dengeleme"] = D
        olay.append(f"PAZARTESI dengeleme: {len(st['baz'])} coin")
    hedef_kaydet(st, D, dict(st["baz"]))
    st["seri"].append([D, round(st["equity"], 6)])
    st["seri"] = st["seri"][-SERI_MAKS:]
    return olay


# ---------------------------------------------------------------- funding carry
def perp_acilis(V, s, d):
    p = V.perp.get(s)
    op = p.acilis(d) if p is not None else float("nan")
    os_ = V.seri[s].acilis(d) if s in V.seri else float("nan")
    if not (np.isfinite(op) and np.isfinite(os_)) or abs(op / os_ - 1) > FND["uyumsuz"]:
        return float("nan")      # funding.py: spot/perp uyumsuz -> gecersiz
    return op


def carry_x(V, s, d):
    """funding.py x[d] = Fd[d] + (Os[d+1]/Os[d]-1) - (Op[d+1]/Op[d]-1)  (notional basina, d acilis -> d+1 acilis)"""
    if V.fund is None or s not in V.fund or s not in V.seri:
        return float("nan")
    fd = V.fund[s].get(d)
    d1 = gun_ekle(d, 1)
    os0, os1 = V.seri[s].acilis(d), V.seri[s].acilis(d1)
    op0, op1 = perp_acilis(V, s, d), perp_acilis(V, s, d1)
    if fd is None or not all(np.isfinite(v) for v in (os0, os1, op0, op1)):
        return float("nan")
    return fd + (os1 / os0 - 1) - (op1 / op0 - 1)


def carry_f7(V, s, d):
    if V.fund is None or s not in V.fund:
        return float("nan")
    v = [V.fund[s].get(gun_ekle(d, -k)) for k in range(7)]
    if any(x is None for x in v):
        return float("nan")
    return float(np.mean(v) * 365)


def funding_esik(acik, f7, E=FND["E"]):
    """funding.w_esik histerezisi: f7 > E ac, f7 < E/2 kapa, arada onceki durum."""
    if f7 > E:
        return True
    if f7 < E / 2:
        return False
    return bool(acik)


def funding_gun(st, V, D):
    if V.fund is None:
        raise RuntimeError("funding verisi alinamadi: " + str(V.hatalar.get("funding", "?")))

    def getiri(s, t):
        x0 = carry_x(V, s, t)                  # I[t+1] var mi
        if not np.isfinite(x0):
            return False, 0.0
        x1 = carry_x(V, s, gun_ekle(t, 1))     # I[t+2]/I[t+1]-1
        return True, float(np.clip(x1 if np.isfinite(x1) else 0.0, -0.95, 5.0))

    portfoy_adim(st, D, getiri, FND["maliyet"])
    U = []
    for s in CEKIRDEK:
        if (np.isfinite(carry_x(V, s, gun_ekle(D, -1))) and np.isfinite(carry_f7(V, s, D))
                and s in V.seri and np.isfinite(V.seri[s].acilis(D)) and np.isfinite(perp_acilis(V, s, D))):
            U.append(s)
    yeni = {}
    for s in CEKIRDEK:
        if s in U and funding_esik(st["acik"].get(s, False), carry_f7(V, s, D)):
            yeni[s] = True
    st["acik"] = yeni
    W = {s: FND["notional"] / len(U) for s in yeni} if U else {}
    hedef_kaydet(st, D, W)
    st["seri"].append([D, round(st["equity"], 6)])
    st["seri"] = st["seri"][-SERI_MAKS:]
    return [f"evren {len(U)} coin, acik {len(yeni)}: {', '.join(s[:-4] for s in yeni) or '-'}; notional %{sum(W.values()) * 100:.1f}"]


# ================================================================ kayit defteri
STRATEJI_FONK = {
    "ema_geri_1d": lambda st, V, D: islem_gun(st, V, D, EMA_CFG),
    "kirilim_1d": lambda st, V, D: islem_gun(st, V, D, KIRILIM_CFG),
    "momentum_haftalik": momentum_gun,
    "tersine_haftalik": tersine_gun,
    "funding_carry": funding_gun,
}
TIP = {"ema_geri_1d": "islem", "kirilim_1d": "islem", "momentum_haftalik": "portfoy",
       "tersine_haftalik": "portfoy", "funding_carry": "portfoy"}


def yeni_strateji(ad):
    b = dict(tip=TIP.get(ad, "portfoy"), baslangic=None, son_gun=None, seri=[], son_hata=None)
    if b["tip"] == "islem":
        b.update(pozisyonlar=[], kapanan=[], toplam_net_r=0.0, islem_sayisi=0, kazanan=0, iptal=0)
    else:
        b.update(equity=1.0, zirve=1.0, dusus=0.0, w_prev={}, hedef={}, baz={}, son_dengeleme=None,
                 islem_sayisi=0, acik={}, rejim=None)
    return b


def islenecek_gunler(st, takvim):
    if not takvim:
        return []
    if st.get("son_gun") is None:
        return takvim[-1:]                      # ilk calisma: sadece son kapanmis gun (gecmis simulasyon YOK)
    return [g for g in takvim if g > st["son_gun"]]


def isle(state, V):
    """Tum stratejileri kacan gunler icin sirayla ilerletir. Return {ad: son gunun olay satirlari}."""
    state.setdefault("surum", 1)
    tum = state.setdefault("stratejiler", {})
    ozet = {}
    for ad, fn in STRATEJI_FONK.items():
        st = tum.get(ad) or yeni_strateji(ad)
        gunler = islenecek_gunler(st, V.takvim)
        if not gunler:
            tum[ad] = st
            ozet[ad] = [f"{st.get('son_gun')} zaten islenmis (degisiklik yok)"]
            continue
        calisma = copy.deepcopy(st)
        try:
            olay = []
            for D in gunler:
                olay = fn(calisma, V, D) or []
                calisma["son_gun"] = D
                if calisma.get("baslangic") is None:
                    calisma["baslangic"] = D
            calisma["son_hata"] = None
            tum[ad] = calisma
            ozet[ad] = olay
        except Exception as e:     # bir stratejinin hatasi digerlerini durdurmaz
            st["son_hata"] = {"gun": V.takvim[-1], "mesaj": f"{type(e).__name__}: {e}"[:300]}
            tum[ad] = st
            ozet[ad] = [f"HATA: {type(e).__name__}: {e}"]
    return ozet


# ================================================================ okuyucular
def json_oku(yol):
    try:
        with open(yol, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def ict_net_r(p):
    """screener.py net_r: gerceklesen R - (maker+taker+kayma)/stop%; dolmayan emirde maliyet yok."""
    r = p.get("realized_r", 0.0) or 0.0
    if p.get("status") == "cancelled" or not p.get("entry_bar_close"):
        return r
    sd = p.get("stop_dist_pct") or ((p.get("risk") or 0) / p["entry"] if p.get("entry") else 0)
    return r - (ICT_MALIYET / sd if sd else 0.0)


def ict_ozet(yol, baslangic_tarih):
    pos = json_oku(yol)
    if not isinstance(pos, list) or not baslangic_tarih:
        return None
    sec = [p for p in pos if str(p.get("opened_at", ""))[:10] >= baslangic_tarih]
    kapali = [p for p in sec if p.get("status") not in ("open", "pending", "cancelled", "active")
              and p.get("entry_bar_close")]
    net = [ict_net_r(p) for p in kapali]
    return dict(n=len(net), toplam=float(sum(net)), ort=float(np.mean(net)) if net else None,
                acik=sum(1 for p in sec if p.get("status") in ("open", "active")),
                bekleyen=sum(1 for p in sec if p.get("status") == "pending"))


TREND_ANAHTAR = (("b3", "b3"), ("yuruyen3", "y3"), ("yuruyen5_tavan", "y5t"))


def okuyucu_guncelle(state, V, trend_yolu):
    ok = state.setdefault("okuyucular", {})
    son = V.takvim[-1]
    btc = V.seri["BTCUSDT"]
    fiyat = float(btc.c[btc.ix[son]])
    ok.setdefault("btc_ref", {"gun": son, "fiyat": fiyat})
    ok["btc_son"] = {"gun": son, "fiyat": fiyat}
    if "trend_ref" not in ok:
        tr = json_oku(trend_yolu)
        pf = (tr or {}).get("portfoy") or {}
        ref = {k: {"equity": pf[k]["equity"], "altut": pf[k]["altut"], "son_gun": pf[k].get("son_gun")}
               for k, _ in TREND_ANAHTAR if isinstance(pf.get(k), dict) and "equity" in pf[k]}
        if ref:
            ok["trend_ref"] = dict(ref, gun=son)


# ================================================================ Telegram raporu
IZINLI_ETIKET = ("<b>", "</b>", "<i>", "</i>", "<pre>", "</pre>")


def _yuzde(x):
    return f"{x * 100:+.1f}%"


def rapor_metni(state, trend_yolu=TREND_STATE, ict_yolu=ICT_FILE):
    """HTML parse_mode: TUM dinamik metin html.escape'ten gecer; sadece IZINLI_ETIKET sabit etiketler kullanilir."""
    E = html.escape
    tum = state.get("stratejiler", {})
    ok = state.get("okuyucular", {})
    son = (ok.get("btc_son") or {}).get("gun", "?")
    tablo = [f"{'strateji':<18}{'getiri':>8}{'islem':>6}{'acik':>6}  not"]
    for ad in STRATEJI_FONK:
        st = tum.get(ad)
        if not st:
            tablo.append(f"{ad:<18}{'-':>8}{'-':>6}{'-':>6}  henuz baslamadi")
            continue
        if st["tip"] == "islem":
            n = st["islem_sayisi"]
            acik = sum(1 for p in st["pozisyonlar"] if p["durum"] == "acik")
            bek = len(st["pozisyonlar"]) - acik
            acik_r = sum(p.get("son_r", 0.0) for p in st["pozisyonlar"] if p["durum"] == "acik")
            notu = (f"ort {st['toplam_net_r'] / n:+.2f}R kaz%{st['kazanan'] / n * 100:.0f}" if n else "islem yok")
            if acik:
                notu += f" acikR {acik_r:+.1f}"
            getiri = f"{st['toplam_net_r']:+.2f}R"
            acik_s = f"{acik}+{bek}" if bek else str(acik)
        else:
            son_w = st["hedef"].get(st.get("son_gun"), {}) if st.get("son_gun") else {}
            getiri = _yuzde(st["equity"] - 1)
            n = st["islem_sayisi"]
            if ad == "momentum_haftalik":
                acik_s = str(sum(1 for v in son_w.values() if v))
                notu = f"rejim {'acik' if st.get('rejim') else 'kapali'} dd%{st['dusus'] * 100:.1f}"
            elif ad == "tersine_haftalik":
                acik_s = f"{sum(1 for v in son_w.values() if v > 0)}L/{sum(1 for v in son_w.values() if v < 0)}S"
                notu = f"dd%{st['dusus'] * 100:.1f}"
            else:
                acik_s = str(len(son_w))
                notu = f"notional%{sum(son_w.values()) * 100:.0f} dd%{st['dusus'] * 100:.1f}"
        if st.get("son_hata"):
            notu = "HATA " + str(st["son_hata"].get("mesaj", ""))[:40]
        tablo.append(f"{ad:<18}{getiri:>8}{n:>6}{acik_s:>6}  {notu}")
    ict = ict_ozet(ict_yolu, state.get("baslangic_tarih"))
    if ict is not None:
        tablo.append(f"{'ict (kagit)':<18}{ict['toplam']:>+7.2f}R{ict['n']:>6}{ict['acik']:>6}  "
                     + (f"ort {ict['ort']:+.2f}R" if ict["n"] else "kapanan yok") + f" bekleyen {ict['bekleyen']}")
    satir = ["<b>ARKA PLAN STRATEJILERI (bilgi amacli, kagit)</b>",
             E(f"Baslangic {state.get('baslangic', '?')} kapanisi -> son gun {son}"),
             "<pre>" + E("\n".join(tablo)) + "</pre>"]
    tr = json_oku(trend_yolu)
    pf = (tr or {}).get("portfoy") or {}
    ref = ok.get("trend_ref") or {}
    parca = []
    for k, et in TREND_ANAHTAR:
        p = pf.get(k)
        if not isinstance(p, dict) or "equity" not in p:
            continue
        if k in ref and ref[k].get("equity"):
            parca.append(f"{et} {_yuzde(p['equity'] / ref[k]['equity'] - 1)} (al-tut {_yuzde(p['altut'] / ref[k]['altut'] - 1)})")
        else:
            parca.append(f"{et} {_yuzde(p['equity'] - 1)} ({p.get('baslangic', '?')}'den beri)")
    satir.append("<b>Trend (demo hesaptaki ana sistem), ayni donem:</b> " + (E(" | ".join(parca)) if parca else "veri yok"))
    br, bs = ok.get("btc_ref"), ok.get("btc_son")
    if br and bs:
        satir.append(E(f"BTC al-tut ayni donem: {_yuzde(bs['fiyat'] / br['fiyat'] - 1)}"))
    satir.append("<i>Bu stratejiler lab testlerinde elendi; ileriye donuk kanit icin izleniyor.</i>")
    return "\n".join(satir)


def rapor_zamani(state, bugun):
    if os.environ.get("ARKAPLAN_HER_GUN") == "1":
        return True
    return pazartesi_mi(bugun) and state.get("son_rapor") != bugun


def tg_send(metin):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[telegram yapilandirilmamis]\n" + metin)
        return False
    d = {"chat_id": chat, "text": metin[:4090], "parse_mode": "HTML"}
    if os.environ.get("TOPIC_SUMMARY"):
        d["message_thread_id"] = os.environ["TOPIC_SUMMARY"]
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data=d, timeout=20)
        if r.status_code != 200:
            print("TG hata:", r.status_code, getattr(r, "text", "")[:200])
        return r.status_code == 200
    except Exception as e:
        print("TG hata:", e)
        return False


# ================================================================ veri toplama
def state_semboller(state):
    out = set()
    for st in (state.get("stratejiler") or {}).values():
        for p in st.get("pozisyonlar", []):
            out.add(p["sym"])
        out |= set(st.get("w_prev", {})) | set(st.get("baz", {}))
        for W in st.get("hedef", {}).values():
            out |= set(W)
    return out


def evren_adaylari(ist):
    ex = ist.get(FAPI + "/fapi/v1/exchangeInfo")
    vadeli = {x["symbol"] for x in ex.get("symbols", [])
              if x.get("quoteAsset") == "USDT" and x.get("contractType") == "PERPETUAL" and x.get("status") == "TRADING"}
    rows = []
    for t in ist.get(SPOT + "/api/v3/ticker/24hr"):
        s = t["symbol"]
        if not s.endswith("USDT") or s not in vadeli:
            continue
        b = s[:-4]
        if any(b.endswith(x) for x in ("UP", "DOWN", "BULL", "BEAR")) or ("USD" in b and len(b) <= 6) or b in HARIC:
            continue
        rows.append((float(t.get("quoteVolume", 0) or 0), s))
    rows.sort(reverse=True)
    return [s for _, s in rows[:ADAY_24S]]


def _adv30(se, g):
    j = se.ix.get(g)
    if j is None or j < 29:
        return float("nan")
    return float(se.qv[j - 29:j + 1].mean())


def veri_topla(ist, state, simdi):
    now_ms = int(simdi.timestamp() * 1000)
    hatalar = {}
    aday = evren_adaylari(ist)
    gerekli = list(dict.fromkeys(["BTCUSDT"] + aday + CEKIRDEK + sorted(state_semboller(state))))

    def al(s):
        try:
            return s, Seri(ist.get(SPOT + "/api/v3/klines", {"symbol": s, "interval": "1d", "limit": KLINE_LIMIT}), now_ms)
        except Exception as e:
            return s, e

    seri = {}
    with ThreadPoolExecutor(ISCI) as ex:
        for s, r in ex.map(al, gerekli):
            if isinstance(r, Exception):
                hatalar[s] = str(r)[:200]
            elif len(r.ot):
                seri[s] = r
    if "BTCUSDT" not in seri:
        raise RuntimeError("BTCUSDT spot verisi alinamadi: " + hatalar.get("BTCUSDT", "?"))
    takvim = seri["BTCUSDT"].gun
    son = takvim[-1]
    skor = [(_adv30(seri[s], son), s) for s in aday if s in seri]
    evren = [s for v, s in sorted([x for x in skor if np.isfinite(x[0])], reverse=True)][:EVREN_N]

    # funding + perp (sadece CEKIRDEK)
    fund, perp = None, {}
    try:
        stf = (state.get("stratejiler") or {}).get("funding_carry") or {}
        ilk = gun_ekle(stf["son_gun"], 1) if stf.get("son_gun") else son
        bas = max(min(gun_ekle(ilk, -10), gun_ekle(son, -10)), gun_ekle(son, -FND["gun"]))
        start_ms = gun_ms(bas) + 60_000       # funding.py: odeme (ts - 60s) gunune yazilir -> bas gunu tam
        limit_k = gun_fark(bas, son) + 3

        def al_f(s):
            try:
                fr = ist.get(FAPI + "/fapi/v1/fundingRate", {"symbol": s, "startTime": start_ms, "limit": 1000})
                d = {}
                for x in fr:
                    g = gun_str(int(x["fundingTime"]) - 60_000)
                    if bas <= g <= son:
                        d[g] = d.get(g, 0.0) + float(x["fundingRate"])
                pk = Seri(ist.get(FAPI + "/fapi/v1/klines", {"symbol": s, "interval": "1d", "limit": limit_k}),
                          now_ms, perp=True)
                return s, (d, pk)
            except Exception as e:
                return s, e

        fund = {}
        with ThreadPoolExecutor(ISCI) as ex:
            for s, r in ex.map(al_f, CEKIRDEK):
                if isinstance(r, Exception):
                    hatalar["funding_" + s] = str(r)[:200]
                else:
                    fund[s], perp[s] = r
        if not fund:
            hatalar["funding"] = "hicbir cekirdek coin icin funding alinamadi"
            fund = None
    except Exception as e:
        hatalar["funding"] = str(e)[:200]
        fund = None
    return Veri(seri, evren, takvim, fund, perp, hatalar)


# ================================================================ ekran ozeti (kuru ve normal kosu)
def detay_satirlari(state, V, ozet):
    D = V.takvim[-1]
    out = [f"== son kapanmis gun {D} | evren {len(V.evren)} coin | veri hatasi {len(V.hatalar)}"]
    tum = state.get("stratejiler", {})
    for ad in STRATEJI_FONK:
        st = tum.get(ad, {})
        out.append(f"\n[{ad}] son_gun={st.get('son_gun')} baslangic={st.get('baslangic')}"
                   + (f" HATA={st['son_hata']}" if st.get("son_hata") else ""))
        out += ["  " + x for x in ozet.get(ad, [])]
        try:
            if st.get("tip") == "islem":
                bek = [p for p in st["pozisyonlar"] if p["durum"] == "bekliyor"]
                acik = [p for p in st["pozisyonlar"] if p["durum"] == "acik"]
                out.append(f"  toplam net {st['toplam_net_r']:+.2f}R, {st['islem_sayisi']} islem | acik {len(acik)} | "
                           f"yarin acilista giris bekleyen {len(bek)}: {', '.join(p['sym'] for p in bek) or '-'}")
                for p in acik:
                    out.append(f"    acik {p['sym']} giris {p['giris']:.6g} ({p['giris_gun']}) stop {p['stop']:.6g} "
                               f"tp {p['tp']:.6g} son_r {p.get('son_r', 0):+.2f}")
            elif ad in ("momentum_haftalik", "tersine_haftalik"):
                W = st.get("hedef", {}).get(D, {})
                out.append(f"  equity {st.get('equity', 1):.4f} | son dengeleme {st.get('son_dengeleme')} | "
                           f"bugunku hedef (yarin acilis) {len(W)} coin, brut %{sum(abs(v) for v in W.values()) * 100:.0f}")
                if not pazartesi_mi(D):
                    bz = momentum_baz(V, D) if ad == "momentum_haftalik" else tersine_baz(V, D)
                    out.append(f"  (bugun pazartesi olsaydi {len(bz)} coin"
                               + (f", BTC rejim={'acik' if btc_rejim(V, D) else 'kapali'}" if ad == "momentum_haftalik" else "")
                               + ")")
                    W = bz if (ad == "tersine_haftalik" or btc_rejim(V, D)) else {}
                    if ad == "momentum_haftalik" and not btc_rejim(V, D):
                        out.append("    rejim kapali -> nakit; rejimsiz baz: " +
                                   ", ".join(f"{s[:-4]} %{v * 100:.1f}" for s, v in sorted(bz.items(), key=lambda x: -x[1])))
                srt = sorted(W.items(), key=lambda x: -x[1])
                if srt:
                    out.append("    long : " + ", ".join(f"{s[:-4]} %{v * 100:.1f}" for s, v in srt if v > 0))
                    if any(v < 0 for _, v in srt):
                        out.append("    short: " + ", ".join(f"{s[:-4]} %{v * 100:.1f}" for s, v in srt[::-1] if v < 0))
            elif ad == "funding_carry":
                out.append(f"  equity {st.get('equity', 1):.5f}")
                parca = []
                for s in CEKIRDEK:
                    f7 = carry_f7(V, s, D)
                    parca.append(f"{s[:-4]} {'-' if not np.isfinite(f7) else f'%{f7 * 100:.1f}'}"
                                 + ("*" if st.get("acik", {}).get(s) else ""))
                out.append("  7g yillik funding (*=acik, esik ac %10 / kapa %5): " + ", ".join(parca))
        except Exception as e:
            out.append(f"  (detay yazilamadi: {e})")
    return out


# ================================================================ ana akis
def state_oku(yol):
    d = json_oku(yol)
    return d if isinstance(d, dict) else {}


def state_yaz(state, yol):
    tmp = yol + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, ensure_ascii=True)
    os.replace(tmp, yol)


def kosu(kuru=False, simdi=None, state_yolu=STATE_FILE, trend_yolu=TREND_STATE, ict_yolu=ICT_FILE):
    simdi = simdi or datetime.now(timezone.utc)
    state = state_oku(state_yolu)
    if kuru:
        state = copy.deepcopy(state)
    ist = Istemci()
    t0 = time.time()
    V = veri_topla(ist, state, simdi)
    bugun = simdi.strftime("%Y-%m-%d")
    state.setdefault("baslangic", V.takvim[-1])
    state.setdefault("baslangic_tarih", bugun)
    ozet = isle(state, V)
    okuyucu_guncelle(state, V, trend_yolu)
    print("\n".join(detay_satirlari(state, V, ozet)))
    if V.hatalar:
        print("\nveri hatalari:", json.dumps(V.hatalar)[:1000])
    print(f"\nistek {ist.say}, sure {time.time() - t0:.0f}s")
    metin = rapor_metni(state, trend_yolu, ict_yolu)
    if kuru:
        print("\n[KURU] state YAZILMADI, Telegram GONDERILMEDI. Rapor metni:\n" + metin)
        return state, metin, False
    gonderildi = False
    if rapor_zamani(state, bugun):
        gonderildi = tg_send(metin)
        if gonderildi:
            state["son_rapor"] = bugun
    state_yaz(state, state_yolu)
    return state, metin, gonderildi


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Arka plan strateji kagit takibi (emir yok)")
    ap.add_argument("--kuru", action="store_true", help="state yazma, Telegram gonderme; bugunku sinyalleri yazdir")
    a = ap.parse_args()
    kosu(kuru=a.kuru)
