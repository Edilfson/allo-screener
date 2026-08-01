"""
Binance Coklu-Strateji Screener v4 + Pozisyon Takibi + Performans Ozeti
========================================================================
Stratejiler (backtest v3'un 5-dilimli sonuclarina gore secildi):
  - ob   [4h, 1d]  : Order Block geri testi  | TP: kosucu | Stop: fib786
  - fvg  [2h, 4h]  : dolmamis FVG geri testi | TP: kosucu | Stop: fib786
  - zone5599 [4h,1d]: EMA55-99 bandi temasi  | TP: klasik | Stop: fib786
Her sinyal strateji etiketiyle kaydedilir -> ozet raporunda strateji bazinda
canli A/B karsilastirmasi yapilir.

TP stilleri:
  - kosucu: fib382'de %50, onceki zirvede %25, 1.618 uzatmada %25
  - klasik: fib382 / zirve / 1.272 uzatma, 1/3-1/3-1/3
Ilk TP sonrasi stop girise (BE) cekilir. Ayni mumda stop+TP: once stop (kotumser).

Telegram Topics: SINYALLER / SONUCLAR / OZET (+ pazartesi icgoru raporu).
UYARI: Yatirim tavsiyesi degildir; sonuclar varsayimsaldir (komisyon/kayma yok).
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf

# ==================== AYARLAR ====================
ALL_INTERVALS = ["2h", "4h", "1d"]
STRATEGY_ORDER = ["ob", "fvg", "zone5599"]          # oncelik sirasi (OB en guclu)
STRATEGY_INTERVALS = {"ob": ["4h", "1d"], "fvg": ["2h", "4h", "1d"], "zone5599": ["4h", "1d"]}
TP_STYLE = {"ob": "kosucu", "fvg": "kosucu", "zone5599": "klasik"}
EMA_PERIODS = [55, 99]

RALLY_MIN_PCT = 0.50
RALLY_MAX_DAYS = 30
PULLBACK_MIN_PCT = 0.05
TOUCH_TOLERANCE_PCT = 0.02
MIN_RR = 2.0
MIN_STOP_DIST_PCT = 0.02      # mutlak alt sinir
MIN_STOP_ATR_MULT = 1.5       # stop en az 1.5*ATR14 uzakta (volatiliteye gore)
MAX_ENTRY_ZONE_POS = 0.60     # bolgenin ust %40indan giris YAPMA (0=dip,1=tepe)
MOVE_STOP_TO_BE = True
# --- bolge kalitesi (OB/FVG) ---
MAX_ZONE_WIDTH_PCT = 0.08     # bolge genisligi fiyatin %8'inden buyukse ele (belirsiz stop)
MIN_FVG_GAP_PCT = 0.005       # %0.5'ten kucuk FVG anlamsiz (gurultu)
# --- iz suren stop (trailing) ---
TRAIL_AFTER_FIRST_TP = True
TRAIL_INTERVALS = {"1d"}      # v4 backtest: trailing 2h/4h'de zarar, 1d'de fayda saglıyor   # ilk TP sonrasi kalan kismi EMA21 ile takip et
TRAIL_EMA = 21
TRAIL_BUFFER = 0.01           # EMA21'in %1 altina koy (fitil payi)

LOOKBACK_CANDLES = 250
# dilim basina mum sayisi: 30 gunluk rally penceresi + EMA99 isinmasi sigmali
LOOKBACK_BY_IV = {"2h": 560, "4h": 320, "1d": 250}
CHART_CANDLES = 120
DEDUP_COOLDOWN_HOURS = 20
POSITION_MAX_DAYS = 45

STATE_FILE = "positions.json"
CHART_DIR = "charts"
BASE_URL = "https://data-api.binance.vision"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC_SIGNALS = os.environ.get("TOPIC_SIGNALS")
TOPIC_RESULTS = os.environ.get("TOPIC_RESULTS")
TOPIC_SUMMARY = os.environ.get("TOPIC_SUMMARY")
SUMMARY_EVERY_RUN = os.environ.get("SUMMARY_EVERY_RUN", "0") == "1"


def bars_per_day(iv):
    return {"1h": 24, "2h": 12, "4h": 6, "12h": 2, "1d": 1}.get(iv, 6)


# ==================== VERI ====================
def get_usdt_symbols():
    r = requests.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    out = []
    for s in r.json()["symbols"]:
        if (s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
                and s.get("isSpotTradingAllowed", True)):
            base = s.get("baseAsset", "")
            if any(base.endswith(x) for x in ("UP", "DOWN", "BULL", "BEAR")):
                continue
            out.append(s["symbol"])
    return sorted(out)


def get_klines(symbol, interval, limit=None, closed_only=False):
    """closed_only=True ise henuz KAPANMAMIS son mum atilir (backtest ile ayni davranis)."""
    if limit is None:
        limit = LOOKBACK_BY_IV.get(interval, LOOKBACK_CANDLES)
    try:
        r = requests.get(f"{BASE_URL}/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=20)
        if r.status_code != 200:
            return None
        raw = r.json()
        if not raw or len(raw) < 60:
            return None
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        if closed_only:
            now = pd.Timestamp.now(tz="UTC")
            df = df[df["close_time"] <= now].reset_index(drop=True)
            if len(df) < 60:
                return None
        return df
    except Exception:
        return None


def add_emas(df):
    df[f"ema{TRAIL_EMA}"] = df["close"].ewm(span=TRAIL_EMA, adjust=False).mean().values
    for p in EMA_PERIODS:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean().values
    return df


# ==================== RALLY + STRATEJI TESPITI ====================
def find_rally(df):
    """Return: (ok, rally_pct, hi, lo, pullback_pct, days, hi_idx, lo_idx)
    hi_idx/lo_idx: df icindeki mutlak konumlar."""
    now_ts = df["close_time"].iloc[-1]
    mask = df["close_time"] >= now_ts - pd.Timedelta(days=RALLY_MAX_DAYS)
    start = int(np.argmax(mask.values))
    closes = df["close"].values[start:]
    times = df["close_time"].values[start:]
    n = len(closes)
    if n < 5:
        return False, 0, 0, 0, 0, 0, 0, 0
    mi = int(np.argmin(closes))
    if mi >= n - 2:
        return False, 0, 0, 0, 0, 0, 0, 0
    ma_i = mi + int(np.argmax(closes[mi:]))
    lo, hi = closes[mi], closes[ma_i]
    if lo <= 0:
        return False, 0, 0, 0, 0, 0, 0, 0
    rpct = (hi - lo) / lo
    days = (pd.Timestamp(times[ma_i]) - pd.Timestamp(times[mi])).total_seconds() / 86400
    if rpct < RALLY_MIN_PCT:
        return False, rpct, hi, lo, 0, days, 0, 0
    pb = (hi - closes[-1]) / hi
    return True, rpct, hi, lo, pb, days, start + ma_i, start + mi


def detect_zone5599(df, hi_idx, lo_idx):
    last = df.iloc[-1]
    a, b = last["ema55"], last["ema99"]
    zl, zh = min(a, b), max(a, b)
    ok = zl * (1 - TOUCH_TOLERANCE_PCT) <= last["close"] <= zh * (1 + TOUCH_TOLERANCE_PCT)
    return ok, (zl, zh)


def detect_fvg(df, hi_idx, lo_idx):
    """Rally bacagindaki dolmamis yukselis FVG'sine (low[j] > high[j-2]) temas."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values[-1]
    i = len(df) - 1
    if hi_idx <= lo_idx + 2 or i <= hi_idx:
        return False, None
    for j in range(hi_idx, lo_idx + 1, -1):
        if j - 2 < 0:
            break
        if l[j] > h[j - 2]:
            zl, zh = h[j - 2], l[j]
            if i > hi_idx + 1 and l[hi_idx + 1:i].min() < zl:
                continue  # bolge zaten dolmus
            if zl <= c <= zh:
                return True, (zl, zh)
    return False, None


def detect_ob(df, hi_idx, lo_idx):
    """Order block: guclu kirilim oncesi son ayi mumunun bolgesine temas."""
    o = df["open"].values
    cl = df["close"].values
    h = df["high"].values
    l = df["low"].values
    c = cl[-1]
    i = len(df) - 1
    if hi_idx <= lo_idx + 2 or i <= hi_idx:
        return False, None
    vol = df["volume"].values
    adaylar = []
    for j in range(hi_idx - 1, max(lo_idx - 3, 1), -1):
        if cl[j] < o[j] and j + 1 <= hi_idx and cl[j + 1] > h[j]:
            zl, zh = l[j], h[j]
            if zl <= 0 or (zh - zl) / zl > MAX_ZONE_WIDTH_PCT:
                continue  # KALITE: asiri genis bolge -> belirsiz stop
            if i > hi_idx + 1 and l[hi_idx + 1:i].min() < zl:
                continue
            if zl <= c <= zh:
                adaylar.append((j, zl, zh))
    if not adaylar:
        return False, None
    # KALITE: hacmi en yuksek OB en guclu kurumsal iz -> onu sec
    ref = vol[max(0, hi_idx - 50):hi_idx + 1].mean() or 1.0
    j, zl, zh = max(adaylar, key=lambda a: vol[a[0]] / ref)
    return True, (zl, zh)


DETECTORS = {"zone5599": detect_zone5599, "fvg": detect_fvg, "ob": detect_ob}


# ==================== PLAN (agirlikli TP) ====================
def compute_trade_plan(swing_low, swing_high, entry, tp_style, zone=None, atr=None):
    """Stop (backtest paritesi + volatilite tabani):
    1) Oncelik fib786 - backtest'te kazanan, genis ve yapisal seviye
    2) fib786 girisin ustundeyse (derin cekilme): bolge alti, sonra swing dip
    3) Her durumda min genislik: max(1.5*ATR14, %2) -> gurultu stoplarini onler"""
    diff = swing_high - swing_low
    if diff <= 0 or entry <= 0:
        return None
    fib786 = swing_high - diff * 0.786
    adaylar = [fib786]                             # 1. tercih: backtest'in kazanan seviyesi
    if zone and zone[0] > 0:
        adaylar.append(zone[0] * 0.998)            # 2. tercih: bolgenin alti
    adaylar.append(swing_low * 0.999)              # 3. tercih: swing dip
    gecerli = [a for a in adaylar if a < entry]
    if not gecerli:
        return None
    stop = gecerli[0]                              # oncelik sirasindaki ILK gecerli seviye

    # VOLATILITE TABANI: gurultuye yem olmayacak genislik
    # (onceki surum sabit %2'ye eziyordu -> stoplarin %56'si tabana yapisip
    #  medyan 6.8 saatte yeniyordu; canli sonuc 152/166 stop)
    min_dist = entry * MIN_STOP_DIST_PCT
    if atr:
        min_dist = max(min_dist, atr * MIN_STOP_ATR_MULT)
    if entry - stop < min_dist:
        stop = entry - min_dist
    risk = entry - stop

    fib382 = swing_high - diff * 0.382
    ext1272 = swing_high + diff * 0.272
    ext1618 = swing_high + diff * 0.618
    if tp_style == "kosucu":
        raw = [(fib382, 0.5), (swing_high, 0.25), (ext1618, 0.25)]
    else:  # klasik
        raw = [(fib382, 1 / 3), (swing_high, 1 / 3), (ext1272, 1 / 3)]

    tps = [(p, w) for p, w in raw if p > entry]
    if not tps:
        return None
    tw = sum(w for _, w in tps)
    tps = sorted([(p, w / tw) for p, w in tps])
    if (tps[0][0] - entry) / risk < MIN_RR:
        return None
    return {"entry": entry, "stop": stop, "risk": risk,
            "tps": [{"p": p, "w": w, "r": (p - entry) / risk, "hit": False} for p, w in tps]}


def format_plan(plan, tp_style):
    lines = [f"🛑 Stop: {plan['stop']:.6g}  (risk %{plan['risk']/plan['entry']*100:.2f}) | TP stili: {tp_style}"]
    for i, tp in enumerate(plan["tps"], 1):
        lines.append(f"🎯 TP{i} (%{tp['w']*100:.0f} pozisyon): {tp['p']:.6g}  → {tp['r']:.1f}R")
    return "\n".join(lines)


# ==================== BAGLAM ====================
def calc_atr(df, period=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(c) < period + 2:
        return None
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    a = tr[:period].mean()
    for i in range(period, len(tr)):
        a = (a * (period - 1) + tr[i]) / period
    return float(a)


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = gains[:period].mean()
    avg_l = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def get_btc_regime():
    df = get_klines("BTCUSDT", "1d")
    if df is None:
        return "bilinmiyor", 0.0
    df = add_emas(df)
    last = df.iloc[-1]
    dist = (last["close"] - last["ema99"]) / last["ema99"]
    return ("boga" if dist >= 0 else "ayi"), float(dist)


def coin_daily_trend(df_1d):
    """Coinin KENDI gunluk trendi: fiyat 1D EMA55/EMA99'a gore nerede?"""
    if df_1d is None or len(df_1d) < 100:
        return "bilinmiyor", 0.0
    last = df_1d.iloc[-1]
    d99 = (last["close"] - last["ema99"]) / last["ema99"]
    if last["close"] >= last["ema55"] and last["close"] >= last["ema99"]:
        t = "yukselis"
    elif last["close"] < last["ema55"] and last["close"] < last["ema99"]:
        t = "dusus"
    else:
        t = "kararsiz"
    return t, float(d99)


def compute_context(df, plan, swing_high, swing_low, btc_regime, btc_dist,
                    coin_trend="bilinmiyor", coin_trend_dist=0.0):
    closes = df["close"].values
    last = df.iloc[-1]
    diff = swing_high - swing_low
    fib_depth = (swing_high - plan["entry"]) / diff if diff > 0 else None
    if fib_depth is not None:
        fib_zone = ("sig(0.382)" if fib_depth < 0.45
                    else "orta(0.5-0.618)" if fib_depth < 0.70 else "derin(0.7+)")
    else:
        fib_zone = "?"
    vol = df["volume"].values
    vol_ratio = float(vol[-1] / vol[-21:-1].mean()) if len(vol) > 21 and vol[-21:-1].mean() > 0 else None
    ema_align = "duzgun(55>99)" if last["ema55"] > last["ema99"] else "ters(55<99)"
    rsi = calc_rsi(closes)
    return {
        "rsi14": round(float(rsi), 1) if rsi is not None else None,
        "vol_ratio": round(float(vol_ratio), 2) if vol_ratio is not None else None,
        "fib_depth": round(float(fib_depth), 3) if fib_depth is not None else None,
        "fib_zone": fib_zone,
        "ema_align": ema_align,
        "btc_regime": btc_regime,
        "btc_ema99_dist": round(btc_dist, 4),
        "coin_1d_trend": coin_trend,
        "coin_1d_ema99_dist": round(coin_trend_dist, 4),
    }


# ==================== POZISYON TAKIBI ====================
def migrate_position(pos):
    """Eski (tp1..r3, 1/3'luk) semayi yeni agirlikli tps semasina cevirir."""
    if "tps" in pos:
        return pos
    tps = []
    for i in (1, 2, 3):
        if f"tp{i}" in pos:
            tps.append({"p": pos[f"tp{i}"], "w": 1 / 3, "r": pos.get(f"r{i}", 0.0),
                        "hit": i in pos.get("tps_hit", [])})
    pos["tps"] = tps
    pos.setdefault("strategy", str(pos.get("ema_period", "eski")))
    return pos


def evaluate_position(pos, df):
    """Acik pozisyonu acilistan sonraki mumlara gore degerlendirir."""
    events = []
    opened = pd.Timestamp(pos["opened_at"])
    # BUG FIX: sadece giristen SONRA ACILAN mumlari degerlendir.
    # Sinyal mumunun kendisi (close_time > opened olsa da) giris oncesi
    # dip/tepe fitillerini icerir; onlari saymak sahte stop/TP uretiyordu.
    ebc = pos.get("entry_bar_close")
    if ebc:
        # yeni pozisyonlar: giris mumunun kapanisindan SONRAKI mumlar (backtest ile ayni)
        future = df[df["close_time"] > pd.Timestamp(ebc)]
    else:
        # eski kayitlar icin geri uyumluluk
        iv_td = pd.Timedelta({"1h": "1h", "2h": "2h", "4h": "4h",
                              "12h": "12h", "1d": "1d"}.get(pos.get("interval", "4h"), "4h"))
        future = df[df["close_time"] >= opened + iv_td]
    if future.empty:
        return pos, events

    entry, risk = pos["entry"], pos["risk"]
    stop = pos["current_stop"]
    tps = pos["tps"]

    for _, cndl in future.iterrows():
        if cndl["low"] <= stop:  # kotumser: once stop
            rem = sum(t["w"] for t in tps if not t["hit"])
            done = sum(t["w"] * t["r"] for t in tps if t["hit"])
            if not any(t["hit"] for t in tps):
                pos["realized_r"] = -1.0
                pos["status"] = "stopped"
                events.append("🛑 STOP oldu (-1.0R)")
            else:
                pos["realized_r"] = done + rem * (stop - entry) / risk
                pos["status"] = "closed_be" if stop >= entry else "stopped"
                nasil = ("iz suren stop" if pos.get("_trailed")
                         else ("BE" if stop >= entry else "stop"))
                events.append(f"🔒 Kalan %{rem*100:.0f} {nasil} ile kapandi "
                              f"(toplam {pos['realized_r']:+.2f}R)")
            pos["closed_at"] = cndl["close_time"].isoformat()
            pos["current_stop"] = stop
            return pos, events

        for i, t in enumerate(tps, 1):
            if not t["hit"] and cndl["high"] >= t["p"]:
                t["hit"] = True
                events.append(f"✅ TP{i} vuruldu ({t['r']:.1f}R, %{t['w']*100:.0f} pozisyon)")
                if MOVE_STOP_TO_BE and stop < entry:
                    stop = entry
                    events.append("🔁 Stop girise (BE) cekildi")

        # IZ SUREN STOP: ilk TP sonrasi kalan kismi EMA21 ile takip et (sadece yukari)
        if (TRAIL_AFTER_FIRST_TP and pos.get("interval") in TRAIL_INTERVALS
                and any(t["hit"] for t in tps)):
            tcol = f"ema{TRAIL_EMA}"
            if tcol in cndl.index and not pd.isna(cndl[tcol]):
                yeni = float(cndl[tcol]) * (1 - TRAIL_BUFFER)
                if yeni > stop:
                    stop = yeni
                    pos["_trailed"] = True

        if all(t["hit"] for t in tps):
            pos["realized_r"] = sum(t["w"] * t["r"] for t in tps)
            pos["status"] = "target_done"
            pos["closed_at"] = cndl["close_time"].isoformat()
            pos["current_stop"] = stop
            events.append(f"🏁 Tum hedefler tamamlandi ({pos['realized_r']:+.2f}R)")
            return pos, events

    pos["current_stop"] = stop
    done = sum(t["w"] * t["r"] for t in tps if t["hit"])
    rem = sum(t["w"] for t in tps if not t["hit"])
    last_close = float(future["close"].iloc[-1])
    pos["unrealized_r"] = done + rem * (last_close - entry) / risk

    age_days = (datetime.now(timezone.utc) - opened.to_pydatetime()).total_seconds() / 86400
    if age_days > POSITION_MAX_DAYS and pos["status"] == "open":
        pos["realized_r"] = pos["unrealized_r"]
        pos["status"] = "timeout"
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        events.append(f"⏳ {POSITION_MAX_DAYS} gun doldu, kapatildi ({pos['realized_r']:+.2f}R)")
    return pos, events


# ==================== OZET / ICGORU ====================
def _grp_line(name, grp):
    r = sum(p.get("realized_r", 0) for p in grp)
    w = len([p for p in grp if p.get("realized_r", 0) > 0])
    return f"  {name}: {len(grp)} islem | {r:+.2f}R | basari %{w/len(grp)*100:.0f}"


def build_summary(positions):
    closed = [p for p in positions if p["status"] != "open"]
    open_ps = [p for p in positions if p["status"] == "open"]
    if not closed:
        return f"📊 <b>OZET</b>\n\nHenuz kapanmis pozisyon yok.\nAcik pozisyon: {len(open_ps)}"

    total_r = sum(p.get("realized_r", 0) for p in closed)
    wins = [p for p in closed if p.get("realized_r", 0) > 0]
    losses = [p for p in closed if p.get("realized_r", 0) <= 0]
    lines = [
        "📊 <b>GENEL OZET</b>",
        f"Kapanan: {len(closed)} islem | Acik: {len(open_ps)}",
        f"Toplam: <b>{total_r:+.2f}R</b> | Islem basi ort: {total_r/len(closed):+.2f}R",
        f"Basari: %{len(wins)/len(closed)*100:.1f} ({len(wins)}K / {len(losses)}Z)",
    ]
    if wins:
        lines.append(f"Ort. kazanc: {sum(p['realized_r'] for p in wins)/len(wins):+.2f}R")
    if losses:
        lines.append(f"Ort. kayip: {sum(p['realized_r'] for p in losses)/len(losses):+.2f}R")

    lines.append("\n<b>Strateji bazinda (canli A/B)</b>")
    for st in sorted(set(p.get("strategy", str(p.get("ema_period", "?"))) for p in closed)):
        g = [p for p in closed if p.get("strategy", str(p.get("ema_period", "?"))) == st]
        if g:
            lines.append(_grp_line(st, g))

    lines.append("\n<b>Zaman dilimi bazinda</b>")
    for iv in ALL_INTERVALS:
        g = [p for p in closed if p["interval"] == iv]
        if g:
            lines.append(_grp_line(iv, g))

    lines.append("\n<b>Kapanis turu</b>")
    for st, lb in [("stopped", "🛑 Stop"), ("closed_be", "🔒 BE/kismi"),
                   ("target_done", "🏁 Tum TP"), ("timeout", "⏳ Zaman asimi")]:
        g = [p for p in closed if p["status"] == st]
        if g:
            lines.append(f"  {lb}: {len(g)} islem | {sum(p.get('realized_r',0) for p in g):+.2f}R")

    best = max(closed, key=lambda p: p.get("realized_r", 0))
    worst = min(closed, key=lambda p: p.get("realized_r", 0))
    lines.append(f"\n🥇 En iyi: {best['symbol']} {best['interval']} {best.get('realized_r',0):+.2f}R")
    lines.append(f"🥶 En kotu: {worst['symbol']} {worst['interval']} {worst.get('realized_r',0):+.2f}R")

    if open_ps:
        lines.append("\n<b>Acik pozisyonlar</b>")
        for p in sorted(open_ps, key=lambda x: x.get("unrealized_r", 0), reverse=True)[:10]:
            hits = "".join(f"✅{i}" for i, t in enumerate(p["tps"], 1) if t["hit"]) or "—"
            lines.append(f"  {p['symbol']} {p['interval']} {p.get('strategy','?')}: "
                         f"{p.get('unrealized_r',0):+.2f}R {hits}")

    lines.append("\n<i>Varsayimsal: agirlikli TP kapanislari, ilk TP sonrasi stop BE. "
                 "Komisyon/kayma dahil degil.</i>")
    return "\n".join(lines)


def build_insights(positions):
    closed = [p for p in positions if p["status"] != "open" and p.get("context")]
    if len(closed) < 5:
        return (f"🔬 <b>ICGORU RAPORU</b>\n\nHenuz yeterli veri yok "
                f"({len(closed)} kapanmis islem, en az 5 gerekli).")
    def bstat(items):
        if not items:
            return None
        r = sum(p.get("realized_r", 0) for p in items)
        w = len([p for p in items if p.get("realized_r", 0) > 0])
        return f"{len(items)} islem | {r:+.2f}R | ort {r/len(items):+.2f}R | basari %{w/len(items)*100:.0f}"
    lines = ["🔬 <b>ICGORU RAPORU</b>", f"(Toplam {len(closed)} kapanmis islem)\n"]
    lines.append("<b>Strateji</b>")
    for st in sorted(set(p.get("strategy", "?") for p in closed)):
        s = bstat([p for p in closed if p.get("strategy") == st])
        if s:
            lines.append(f"  {st}: {s}")
    lines.append("\n<b>Fib bolgesi</b>")
    for z in ["sig(0.382)", "orta(0.5-0.618)", "derin(0.7+)"]:
        s = bstat([p for p in closed if p["context"].get("fib_zone") == z])
        if s:
            lines.append(f"  {z}: {s}")
    lines.append("\n<b>Coinin 1D trendi</b>")
    for tr in ["yukselis", "kararsiz", "dusus"]:
        s2 = bstat([p for p in closed if p["context"].get("coin_1d_trend") == tr])
        if s2:
            lines.append(f"  {tr}: {s2}")
    lines.append("\n<b>BTC rejimi</b>")
    for rg in ["boga", "ayi"]:
        s = bstat([p for p in closed if p["context"].get("btc_regime") == rg])
        if s:
            lines.append(f"  {rg}: {s}")
    lines.append("\n<b>RSI(14)</b>")
    for lb, lo_r, hi_r in [("asiri satim <35", 0, 35), ("notr 35-55", 35, 55), ("guclu 55+", 55, 101)]:
        g = [p for p in closed if p["context"].get("rsi14") is not None
             and lo_r <= p["context"]["rsi14"] < hi_r]
        s = bstat(g)
        if s:
            lines.append(f"  {lb}: {s}")
    lines.append("\n<i>20'den az islemli gruplardan sonuc cikarma; bu rapor oneri degil gozlemdir.</i>")
    return "\n".join(lines)


# ==================== GRAFIK ====================
def make_chart(df, symbol, interval, strategy, plan, zone=None,
               rally=None):
    """Zenginlestirilmis grafik: golgeli bolge + rally bacagi + R etiketli seviyeler."""
    os.makedirs(CHART_DIR, exist_ok=True)
    d = df.tail(CHART_CANDLES).copy()
    d = d.set_index(pd.DatetimeIndex(d["close_time"]))
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                          "close": "Close", "volume": "Volume"})
    aps = [mpf.make_addplot(d[f"ema{p}"], color={55: "orange", 99: "purple"}[p], width=1.1)
           for p in EMA_PERIODS]

    prices = [plan["stop"], plan["entry"]] + [t["p"] for t in plan["tps"]]
    colors = ["#ff5252", "#ffffff", "#90ee90", "#2ecc71", "#f1c40f"][:len(prices)]
    kw = {}
    if zone:  # OB/FVG bolgesi: golgeli dikdortgen
        kw["fill_between"] = dict(y1=float(zone[0]), y2=float(zone[1]),
                                  alpha=0.18, color="#00bcd4")
    # rally bacagi: dip -> zirve cizgisi
    if rally:
        lo_t, lo_p, hi_t, hi_p = rally
        if d.index[0] <= lo_t <= d.index[-1] and d.index[0] <= hi_t <= d.index[-1]:
            kw["alines"] = dict(alines=[[(lo_t, lo_p), (hi_t, hi_p)]],
                                colors=["#8e9aaf"], linestyle="-.", linewidths=1.0)

    path = os.path.join(CHART_DIR, f"{symbol}_{interval}.png")
    fig, axes = mpf.plot(
        d, type="candle", style="binance", addplot=aps, volume=True,
        title=f"{symbol} - {interval}  [{strategy.upper()}]",
        hlines=dict(hlines=prices, colors=colors, linestyle="--", linewidths=1.1),
        returnfig=True, figsize=(12, 7), **kw)

    ax = axes[0]
    etiketler = [("STOP", plan["stop"], colors[0]), ("GIRIS", plan["entry"], colors[1])]
    for i, t in enumerate(plan["tps"]):
        etiketler.append((f"TP{i+1} %{t['w']*100:.0f} ({t['r']:.1f}R)", t["p"], colors[2 + i]))
    xmax = len(d) - 1
    for lbl, y, col in etiketler:
        ax.text(xmax * 1.005, y, f" {lbl} {y:.6g}", color=col, fontsize=8,
                va="center", ha="left", family="monospace")
    if zone:
        ax.text(xmax * 1.005, (zone[0] + zone[1]) / 2, " BOLGE", color="#00bcd4",
                fontsize=8, va="center", ha="left", family="monospace")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return path


# ==================== TELEGRAM ====================
def tg_send(text, topic=None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG yok]", text[:300]); return False
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:4090], "parse_mode": "HTML"}
    if topic:
        data["message_thread_id"] = topic
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data=data, timeout=20)
        if r.status_code != 200:
            print("TG hata:", r.text[:200])
        return r.status_code == 200
    except Exception as e:
        print("TG hata:", e); return False


def tg_photo(path, caption, topic=None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG yok - foto]", path); return False
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000], "parse_mode": "HTML"}
    if topic:
        data["message_thread_id"] = topic
    try:
        with open(path, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                              data=data, files={"photo": f}, timeout=45)
        if r.status_code != 200:
            print("TG foto hata:", r.text[:200])
        return r.status_code == 200
    except Exception as e:
        print("TG foto hata:", e); return False


# ==================== STATE ====================
def load_positions():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                ps = json.load(f)
            return [migrate_position(p) for p in ps]
        except Exception:
            return []
    return []


def save_positions(ps):
    with open(STATE_FILE, "w") as f:
        json.dump(ps, f, indent=1)


# ==================== ANA AKIS ====================
def main():
    positions = load_positions()
    now = datetime.now(timezone.utc)

    open_ps = [p for p in positions if p["status"] == "open"]
    print(f"{len(open_ps)} acik pozisyon kontrol ediliyor...")
    for pos in open_ps:
        df = get_klines(pos["symbol"], pos["interval"])
        if df is None:
            continue
        df = add_emas(df)          # iz suren stop EMA21'e ihtiyac duyar
        _, events = evaluate_position(pos, df)
        if events:
            head = (f"\U0001F4CC <b>{pos['symbol']}</b> {pos['interval']} [{pos.get('strategy','?')}]\n"
                    f"Giris: {pos['entry']:.6g} | Acilis: {pos['opened_at'][:10]}")
            tg_send(head + "\n" + "\n".join(events), TOPIC_RESULTS)
        time.sleep(0.05)

    symbols = get_usdt_symbols()
    print(f"{len(symbols)} sembol taranacak...")
    btc_regime, btc_dist = get_btc_regime()
    print(f"BTC rejimi: {btc_regime} (%{btc_dist*100:.1f})")
    new_count = 0

    # tani sayaclari: OB/FVG nicin gelmiyor sorusuna log'dan cevap
    diag_rally = {iv: 0 for iv in ALL_INTERVALS}
    diag_detect = {st: 0 for st in STRATEGY_ORDER}
    diag_plan_red = {st: 0 for st in STRATEGY_ORDER}
    diag_konum = {st: 0 for st in STRATEGY_ORDER}

    for symbol in symbols:
        try:
            # strateji BASINA acik pozisyon / cooldown (gercek A/B icin bagimsiz)
            open_strats = {p.get("strategy") for p in positions
                           if p["symbol"] == symbol and p["status"] == "open"}
            last_by_strat = {}
            for p in positions:
                if p["symbol"] == symbol:
                    st = p.get("strategy", "?")
                    ts = pd.Timestamp(p["opened_at"])
                    if st not in last_by_strat or ts > last_by_strat[st]:
                        last_by_strat[st] = ts

            dfs, rallies = {}, {}
            for iv in ALL_INTERVALS:
                df = get_klines(symbol, iv, closed_only=True)   # sadece KAPANMIS mumlar
                dfs[iv] = add_emas(df) if df is not None else None
                if dfs[iv] is not None:
                    rallies[iv] = find_rally(dfs[iv])
                    ok, _, _, _, pb, _, _, _ = rallies[iv]
                    if ok and pb >= PULLBACK_MIN_PCT:
                        diag_rally[iv] += 1
                time.sleep(0.04)

            for strat in STRATEGY_ORDER:
                if strat in open_strats:
                    continue
                lb = last_by_strat.get(strat)
                if lb is not None and (now - lb.to_pydatetime()).total_seconds() / 3600 < DEDUP_COOLDOWN_HOURS:
                    continue

                trig = None
                for iv in STRATEGY_INTERVALS[strat]:
                    if dfs.get(iv) is None or iv not in rallies:
                        continue
                    ok, rpct, hi, lo, pb, days, hi_idx, lo_idx = rallies[iv]
                    if not ok or pb < PULLBACK_MIN_PCT:
                        continue
                    hit, zone = DETECTORS[strat](dfs[iv], hi_idx, lo_idx)
                    if not hit:
                        continue
                    diag_detect[strat] += 1
                    # GIRIS KONUMU: bolgenin ust kismindan girme (kotu fiyat + dar stop)
                    if strat in ("ob", "fvg") and zone and zone[1] > zone[0]:
                        konum = (float(dfs[iv].iloc[-1]["close"]) - zone[0]) / (zone[1] - zone[0])
                        if konum > MAX_ENTRY_ZONE_POS:
                            diag_konum[strat] += 1
                            continue
                    plan = compute_trade_plan(lo, hi, float(dfs[iv].iloc[-1]["close"]),
                                              TP_STYLE[strat], zone=zone,
                                              atr=calc_atr(dfs[iv]))
                    if not plan:
                        diag_plan_red[strat] += 1
                        continue
                    trig = (iv, rpct, hi, lo, pb, days, plan, zone, hi_idx, lo_idx)
                    break

                if not trig:
                    continue

                iv, rpct, hi, lo, pb, days, plan, zone, hi_idx, lo_idx = trig
                c_trend, c_dist = coin_daily_trend(dfs.get("1d"))
                ctx = compute_context(dfs[iv], plan, hi, lo, btc_regime, btc_dist,
                                      c_trend, c_dist)

                last = dfs[iv].iloc[-1]
                body = abs(last["close"] - last["open"])
                lower_wick = min(last["close"], last["open"]) - last["low"]
                q_ok = (body > 0 and lower_wick >= body) or last["close"] >= max(last["ema55"], last["ema99"])
                quality_line = "Kalite: " + ("\u2705 fitil/band ustu" if q_ok else "\u26A0\uFE0F zayif mum")

                ctx_line = (f"\n\n\U0001F9ED RSI:{ctx['rsi14']} | Hacim:{ctx['vol_ratio']}x | "
                            f"Fib:{ctx['fib_zone']} | {ctx['ema_align']} | BTC:{ctx['btc_regime']}\n"
                            f"1D trend: {ctx['coin_1d_trend']}"
                            + (" \u26A0\uFE0F gunluk dususte - karsi yonde islem" if ctx['coin_1d_trend'] == 'dusus' else "") + "\n"
                            f"{quality_line} | Onay: sonraki yesil kapanis hafif avantajli (bilgi)")

                strat_lbl = {"ob": "ORDER BLOCK", "fvg": "FVG", "zone5599": "EMA55-99 bolgesi"}[strat]
                zone_line = ""
                if strat in ("ob", "fvg") and zone:
                    zone_line = f"Bolge: {zone[0]:.6g} - {zone[1]:.6g}\n"
                msg = (f"\U0001F514 <b>{symbol}</b>  [{iv} / {strat_lbl}]\n"
                       f"Yukselis: %{rpct*100:.1f} ({days:.1f} gunde)\n"
                       f"Zirve {hi:.6g} \u2192 simdi {plan['entry']:.6g} (%{pb*100:.1f} geri cekildi)\n"
                       + zone_line + "\n"
                       f"\U0001F4CB <b>Islem Plani</b>\n" + format_plan(plan, TP_STYLE[strat])
                       + ctx_line)

                sent = False
                try:
                    _d = dfs[iv]
                    rally_ln = (_d["close_time"].iloc[lo_idx], float(lo),
                                _d["close_time"].iloc[hi_idx], float(hi)) \
                        if 0 <= lo_idx < len(_d) and 0 <= hi_idx < len(_d) else None
                    chart = make_chart(_d, symbol, iv, strat, plan,
                                       zone=zone if strat in ("ob", "fvg") else None,
                                       rally=rally_ln)
                    sent = tg_photo(chart, msg, TOPIC_SIGNALS)
                except Exception as e:
                    print(f"{symbol} grafik hatasi: {e}")
                if not sent:
                    tg_send(msg, TOPIC_SIGNALS)

                positions.append({
                    "symbol": symbol, "interval": iv, "strategy": strat,
                    "ema_period": strat,
                    "opened_at": now.isoformat(), "status": "open",
                    "entry": plan["entry"], "stop": plan["stop"], "current_stop": plan["stop"],
                    "risk": plan["risk"], "tps": plan["tps"],
                    "entry_bar_close": dfs[iv].iloc[-1]["close_time"].isoformat(),
                    # --- analiz icin ek alanlar (sonradan iyilestirme yapabilmek icin) ---
                    "tp_style": TP_STYLE[strat],
                    "zone_low": float(zone[0]) if zone else None,
                    "zone_high": float(zone[1]) if zone else None,
                    "zone_width_pct": (round((zone[1] - zone[0]) / zone[0], 4)
                                       if zone and zone[0] else None),
                    "quality_ok": bool(q_ok),
                    "swing_high": float(hi), "swing_low": float(lo),
                    "stop_dist_pct": round(plan["risk"] / plan["entry"], 4),
                    "nearest_tp_r": round(min(t["r"] for t in plan["tps"]), 2),
                    "scan_version": "v6",
                    "rally_pct": rpct, "rally_days": days, "context": ctx,
                    "realized_r": 0.0, "unrealized_r": 0.0,
                })
                new_count += 1

        except Exception as e:
            print(f"{symbol} hata: {e}")
            continue

    print(f"{new_count} yeni sinyal.")
    print(f"TANI | rally gecen (sembol/dilim): {diag_rally}")
    print(f"TANI | bolge tespiti: {diag_detect} | konum reddi: {diag_konum} | plan reddi: {diag_plan_red}")

    if SUMMARY_EVERY_RUN or now.hour < 4:
        tg_send(build_summary(positions), TOPIC_SUMMARY)
        if SUMMARY_EVERY_RUN or now.weekday() == 0:
            tg_send(build_insights(positions), TOPIC_SUMMARY)

    save_positions(positions)
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
