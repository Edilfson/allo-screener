"""
RANGE + YAPI REDDI + LIKIDITE HEDEFI kurulumu (R1-R7)
======================================================
SHORT: lower high (LH) | LONG: higher low (HL)
Spesifikasyon: kurulum_spesifikasyonu.md (R1-R7 birebir uygulanir)

Canli screener icin tespit modulu. Backtest karsiligi: backtest_range.py

KRITIK: swing noktalari k bar GECIKMEYLE teyit edilir (ileriye bakma yok).
Kapali mumlarla calisilmalidir (screener.get_klines(closed_only=True)).

UYARI: Bu kurulumun karli oldugu DOGRULANMAMISTIR. Once backtest_range.py
ile out-of-sample dogrulama yapin. Short sinyalleri spot'ta uygulanamaz,
vadeli islem gerektirir ve funding maliyeti dogurur.
"""

import numpy as np

# --- R1-R7 parametreleri (spesifikasyondaki varsayilanlar) ---
SWING_K = 3
LOOKBACK = 120
MIN_RANGE_PCT = 0.03
MAX_RANGE_PCT = 0.15
TREND_MAX = 0.06
ZONE_PCT = 0.40
STOP_BUF = 0.002
TARGET_BUF = 0.002
MIN_RR = 3.0
MAX_CONF_COST = 0.25       # R7: 1/(3+1) -> 1:3 icin %25


def swing_flags(high, low, k=SWING_K):
    """Fraktal swing: i noktasi, +-k penceresinin tepesi/dibi ise swing'dir.
    i noktasi ancak i+k barinda TEYITLENIR (ileriye bakma korumasi)."""
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        wh = high[i - k:i + k + 1]
        if high[i] == wh.max() and wh.argmax() == k:
            sh[i] = True
        wl = low[i - k:i + k + 1]
        if low[i] == wl.min() and wl.argmin() == k:
            sl[i] = True
    return sh, sl


def detect_range_setup(df, side):
    """side: -1 SHORT (lower high) | +1 LONG (higher low)
    Son KAPALI mumda kurulum var mi? Return: (plan dict | None, sebep)."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)
    i = n - 1                                   # son kapali mum
    if n < LOOKBACK + SWING_K + 5:
        return None, "veri_yetersiz"

    # --- R1: range var mi ---
    seg_h = high[i - LOOKBACK:i + 1]
    seg_l = low[i - LOOKBACK:i + 1]
    range_high, range_low = seg_h.max(), seg_l.min()
    rng = range_high - range_low
    if rng <= 0 or range_low <= 0:
        return None, "range_gecersiz"
    range_pct = rng / range_low
    if not (MIN_RANGE_PCT <= range_pct <= MAX_RANGE_PCT):
        return None, "range"

    # --- R2: trend filtresi (guclu trendde mean-reversion alma) ---
    drift = abs(close[i] - close[i - LOOKBACK]) / close[i - LOOKBACK]
    if drift > TREND_MAX:
        return None, "trend"

    # --- R3: yapi (LH / HL), k bar gecikmeyle TEYITLI swingler ---
    sh, sl = swing_flags(high, low, SWING_K)
    flags = sh if side == -1 else sl
    px = high if side == -1 else low
    idxs = np.flatnonzero(flags)
    conf = idxs[idxs <= i - SWING_K]             # ILERIYE BAKMA KORUMASI
    if len(conf) < 2:
        return None, "swing_yok"
    s_i, prev_i = conf[-1], conf[-2]
    if s_i < i - LOOKBACK:
        return None, "swing_eski"

    # short: yeni tepe daha DUSUK | long: yeni dip daha YUKSEK
    if side * (px[s_i] - px[prev_i]) <= 0:
        return None, "yapi_yok"

    # konum: short -> range ust %40, long -> alt %40
    if side == -1:
        in_zone = px[s_i] >= range_high - ZONE_PCT * rng
    else:
        in_zone = px[s_i] <= range_low + ZONE_PCT * rng
    if not in_zone:
        return None, "bolge_disi"

    # --- R4: teyit (yapi barinin ters ucunu kapanisla kir, YENI olmali) ---
    trig = low[s_i] if side == -1 else high[s_i]
    if side == -1:
        broke_now, broke_prev = close[i] < trig, close[i - 1] < trig
    else:
        broke_now, broke_prev = close[i] > trig, close[i - 1] > trig
    if not (broke_now and not broke_prev):
        return None, "teyit_yok"

    # --- R5: seviyeler ---
    entry = float(close[i])
    if side == -1:
        s_px = max(px[prev_i], px[s_i])
        stop = s_px * (1 + STOP_BUF)
        target = range_low * (1 + TARGET_BUF)    # likidite havuzunun HEMEN USTU
    else:
        s_px = min(px[prev_i], px[s_i])
        stop = s_px * (1 - STOP_BUF)
        target = range_high * (1 - TARGET_BUF)   # havuzun HEMEN ALTI
    risk = side * (entry - stop)
    reward = side * (target - entry)
    if risk <= 0 or reward <= 0:
        return None, "seviye_gecersiz"

    # --- R7: teyit maliyeti (spesifikasyonun ozgun katkisi) ---
    struct_h = abs(s_px - target)
    conf_cost = abs(s_px - entry) / struct_h if struct_h > 0 else 9.9
    if conf_cost > MAX_CONF_COST:
        return None, "teyit_maliyeti"

    # --- R6: R:R ---
    rr = reward / risk
    if rr < MIN_RR:
        return None, "rr"

    return {
        "side": int(side),
        "entry": entry,
        "stop": float(stop),
        "risk": float(risk),
        "tps": [{"p": float(target), "w": 1.0, "r": float(rr), "hit": False}],
        "range_high": float(range_high),
        "range_low": float(range_low),
        "range_pct": round(float(range_pct), 4),
        "struct_price": float(s_px),
        "conf_cost": round(float(conf_cost), 4),
        "drift": round(float(drift), 4),
    }, "PASS"
