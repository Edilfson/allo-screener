"""ICT / SMC CANLI TESPIT MODULU
Kurallar: BOS + displacement(1.5xATR) + FVG esligi + Order Block + mitigation
Giris: OB %50 LIMIT | Stop: OB uzak ucu | TP: giris + 5R (TEK)
Backtest 2h SHORT: IS +0.336R -> OOS +0.304R
UYARI: kontrol testi yapilmadi, yatirim tavsiyesi degildir.
"""

import numpy as np

SWING_K = 3
DISP_MULT = 1.5
MIN_STOP_PCT = 0.006   # canli veri: %0.5 tabanindaki 6 islem 0.0R verdi (gurultu)
MAX_STOP_PCT = 0.05    # canli veri: %5+ stoplu 2 islem de kaybetti (bolge belirsiz)
TP_R = 5.0
LOOKBACK_BOS = 40
OB_ARAMA = 12
MAX_ENTRY_DIST_R = 1.5   # fiyat limit girise en fazla bu kadar RISK uzakta olabilir
                         # (backtestte limit 10 mum icinde doluyordu; canli taramada
                         #  bolge eski olabilir ve fiyat cok uzaklasmis olabilir)
MAX_BOS_YAS = 20         # BOS bu kadar mumdan eskiyse kurulum bayat sayilir


def _atr(h, l, c, period=14):
    n = len(c)
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    out = np.full(n, np.nan)
    if len(tr) <= period:
        return out
    a = tr[:period].mean()
    out[period] = a
    for i in range(period + 1, n):
        a = (a * (period - 1) + tr[i - 1]) / period
        out[i] = a
    return out


def _swings(h, l, k=SWING_K):
    n = len(h)
    sh = np.zeros(n, bool)
    sl = np.zeros(n, bool)
    for i in range(k, n - k):
        w = h[i - k:i + k + 1]
        if h[i] == w.max() and w.argmax() == k:
            sh[i] = True
        w = l[i - k:i + k + 1]
        if l[i] == w.min() and w.argmin() == k:
            sl[i] = True
    return sh, sl


def detect_ict(df, side):
    """side: +1 LONG | -1 SHORT. Return: (plan|None, sebep)"""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(c)
    i = n - 1
    if n < 80:
        return None, "veri_yetersiz"
    atr = _atr(h, l, c)
    if np.isnan(atr[i]):
        return None, "atr_yok"
    sh, sl = _swings(h, l)
    flags = sh if side == 1 else sl
    px = h if side == 1 else l
    conf = np.flatnonzero(flags[:max(0, i - SWING_K)])
    if len(conf) < 2:
        return None, "swing_yok"
    swing_seviye = px[conf[-1]]

    bos_i = None
    for j in range(max(0, i - LOOKBACK_BOS), i):
        if j <= conf[-1]:
            continue
        if (c[j] > swing_seviye) if side == 1 else (c[j] < swing_seviye):
            bos_i = j
            break
    if bos_i is None:
        return None, "bos_yok"
    if abs(c[bos_i] - o[bos_i]) < DISP_MULT * atr[bos_i]:
        return None, "displacement_zayif"

    fvg = None
    for j in range(max(bos_i - 2, 2), min(bos_i + 4, n)):
        if side == 1 and l[j] > h[j - 2]:
            fvg = (float(h[j - 2]), float(l[j]))
            break
        if side == -1 and h[j] < l[j - 2]:
            fvg = (float(h[j]), float(l[j - 2]))
            break
    if fvg is None:
        return None, "fvg_yok"

    ob = None
    for j in range(bos_i, max(bos_i - OB_ARAMA, 0), -1):
        if side == 1 and c[j] < o[j]:
            ob = (float(l[j]), float(h[j]))
            break
        if side == -1 and c[j] > o[j]:
            ob = (float(l[j]), float(h[j]))
            break
    if ob is None:
        return None, "ob_yok"

    if i > bos_i + 1:
        if side == 1 and l[bos_i + 1:i].min() < ob[0]:
            return None, "bolge_dolmus"
        if side == -1 and h[bos_i + 1:i].max() > ob[1]:
            return None, "bolge_dolmus"

    lo, hi = ob
    entry = (lo + hi) / 2.0
    if side == 1:
        stop = lo * 0.999
        if stop >= entry or (entry - stop) / entry < MIN_STOP_PCT:
            stop = entry * (1 - MIN_STOP_PCT)
        risk = entry - stop
    else:
        stop = hi * 1.001
        if stop <= entry or (stop - entry) / entry < MIN_STOP_PCT:
            stop = entry * (1 + MIN_STOP_PCT)
        risk = stop - entry
    if risk <= 0:
        return None, "seviye_gecersiz"

    # STOP BANDI: canli sonuclara gore kazanan bant %0.6 - %5
    stop_pct = risk / entry
    if stop_pct < MIN_STOP_PCT:
        return None, "stop_cok_dar"
    if stop_pct > MAX_STOP_PCT:
        return None, "stop_cok_genis"

    tp = entry + side * TP_R * risk
    son_fiyat = float(c[i])

    # TAZELIK 1: BOS cok eskiyse kurulum bayat
    if (i - bos_i) > MAX_BOS_YAS:
        return None, "bayat_kurulum"

    # TAZELIK 2: fiyat limit girise cok uzaksa emir yakin zamanda dolmaz
    mesafe_r = abs(entry - son_fiyat) / risk
    if mesafe_r > MAX_ENTRY_DIST_R:
        return None, "giris_cok_uzak"

    # YON: long ise fiyat girisin USTUNDE olmali (bolgeye inmesi beklenir)
    #      short ise fiyat girisin ALTINDA olmali (bolgeye cikmasi beklenir)
    if side == 1 and son_fiyat < entry:
        return None, "bolge_gecilmis"
    if side == -1 and son_fiyat > entry:
        return None, "bolge_gecilmis"
    if side == 1 and son_fiyat < stop:
        return None, "kacirilmis"
    if side == -1 and son_fiyat > stop:
        return None, "kacirilmis"

    return {
        "side": int(side),
        "entry": float(entry),
        "stop": float(stop),
        "risk": float(risk),
        "tps": [{"p": float(tp), "w": 1.0, "r": float(TP_R), "hit": False}],
        "ob_low": lo, "ob_high": hi,
        "fvg_low": fvg[0], "fvg_high": fvg[1],
        "swing": float(swing_seviye),
        "bos_bar": int(bos_i),
        "son_fiyat": son_fiyat,
        "limit_mesafe_pct": round((entry - son_fiyat) / son_fiyat * 100, 2),
    }, "PASS"
