

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
INTERVALS = ["4h", "1d"]
EMA_PERIODS = [55, 99]

RALLY_MIN_PCT = 0.50          # min %50 yukselis
RALLY_MAX_DAYS = 30           # yukselis son kac gun icinde olmali
PULLBACK_MIN_PCT = 0.05       # zirveden min geri cekilme
TOUCH_TOLERANCE_PCT = 0.015   # EMA'ya bu mesafe = temas
MIN_RR = 3.0                  # minimum odul/risk orani
MIN_STOP_DIST_PCT = 0.02      # stop girise en az bu kadar uzak olmali (gurultu korumasi)
MOVE_STOP_TO_BE_AFTER_TP1 = True

LOOKBACK_CANDLES = 250
CHART_CANDLES = 120
DEDUP_COOLDOWN_HOURS = 20
POSITION_MAX_DAYS = 45        # bu suredan sonra hala acik pozisyon zaman asimiyla kapanir

STATE_FILE = "positions.json"
CHART_DIR = "charts"
BASE_URL = "https://data-api.binance.vision"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC_SIGNALS = os.environ.get("TOPIC_SIGNALS")
TOPIC_RESULTS = os.environ.get("TOPIC_RESULTS")
TOPIC_SUMMARY = os.environ.get("TOPIC_SUMMARY")
SUMMARY_EVERY_RUN = os.environ.get("SUMMARY_EVERY_RUN", "0") == "1"


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
                continue  # kaldiracli tokenlar yaniltici
            out.append(s["symbol"])
    return sorted(out)


def get_klines(symbol, interval, limit=LOOKBACK_CANDLES):
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
        return df
    except Exception:
        return None


def add_emas(df):
    for p in EMA_PERIODS:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


# ==================== STRATEJI ====================
def find_rally(df):
    """Return: (ok, rally_pct, swing_high, swing_low, pullback_pct, rally_days)"""
    now_ts = df["close_time"].iloc[-1]
    win = df[df["close_time"] >= now_ts - pd.Timedelta(days=RALLY_MAX_DAYS)].reset_index(drop=True)
    closes, times = win["close"].values, win["close_time"].values
    n = len(closes)
    if n < 5:
        return False, 0, 0, 0, 0, 0

    min_idx = int(np.argmin(closes))
    if min_idx >= n - 2:
        return False, 0, 0, 0, 0, 0
    max_idx = min_idx + int(np.argmax(closes[min_idx:]))
    lo, hi = closes[min_idx], closes[max_idx]
    if lo <= 0:
        return False, 0, 0, 0, 0, 0

    rally_pct = (hi - lo) / lo
    days = (pd.Timestamp(times[max_idx]) - pd.Timestamp(times[min_idx])).total_seconds() / 86400
    if rally_pct < RALLY_MIN_PCT:
        return False, rally_pct, hi, lo, 0, days
    return True, rally_pct, hi, lo, (hi - closes[-1]) / hi, days


def ema_distances(dfs):
    rows = []
    for interval, df in dfs.items():
        if df is None:
            continue
        last = df.iloc[-1]
        for p in EMA_PERIODS:
            ema = last[f"ema{p}"]
            rows.append({"interval": interval, "ema_period": p, "ema_val": ema,
                         "dist_pct": abs(last["close"] - ema) / ema})
    return rows


def format_ema_lines(rows):
    if not rows:
        return ""
    closest = min(range(len(rows)), key=lambda i: rows[i]["dist_pct"])
    return "\n".join(
        f"{'🎯' if i == closest else '📍'} {r['interval']} EMA{r['ema_period']}: "
        f"{r['ema_val']:.6g} (mesafe %{r['dist_pct']*100:.2f})"
        for i, r in enumerate(rows))


def compute_trade_plan(swing_low, swing_high, entry):
    """Fib bazli 3 TP + yapisal stop. MIN_RR saglanamiyorsa None -> sinyal atlanir."""
    diff = swing_high - swing_low
    if diff <= 0 or entry <= 0:
        return None

    tp1 = swing_high - diff * 0.382     # orta direnc
    tp2 = swing_high                    # onceki zirve
    tp3 = swing_high + diff * 0.272     # devam / uzatma
    fib_786 = swing_high - diff * 0.786

    stop = max(fib_786, swing_low * 0.999)
    if stop >= entry:
        return None

    # stop cok dar kalmasin (gurultuye yem olmasin)
    if (entry - stop) / entry < MIN_STOP_DIST_PCT:
        stop = entry * (1 - MIN_STOP_DIST_PCT)

    risk = entry - stop
    if risk <= 0:
        return None

    tps = [t for t in (tp1, tp2, tp3) if t > entry]
    if not tps:
        return None

    # MIN_RR filtresi: yeterli R yoksa bu kurulumu hic alma
    if (min(tps) - entry) / risk < MIN_RR:
        return None

    return {"entry": entry, "stop": stop, "risk": risk,
            "tp1": tp1, "r1": (tp1 - entry) / risk,
            "tp2": tp2, "r2": (tp2 - entry) / risk,
            "tp3": tp3, "r3": (tp3 - entry) / risk}


def format_plan(plan):
    lines = [f"🛑 Stop: {plan['stop']:.6g}  (risk %{plan['risk']/plan['entry']*100:.2f})"]
    for lbl, tp, r in [("TP1 orta direnc", plan["tp1"], plan["r1"]),
                       ("TP2 onceki zirve", plan["tp2"], plan["r2"]),
                       ("TP3 devam", plan["tp3"], plan["r3"])]:
        if tp > plan["entry"]:
            lines.append(f"🎯 {lbl}: {tp:.6g}  → {r:.1f}R")
    return "\n".join(lines)


# ==================== POZISYON TAKIBI ====================
def evaluate_position(pos, df):
    """Acik pozisyonu acilistan sonraki mumlara gore degerlendirir.
    Return: (pos, events) - events bu turda yeni gerceklesen olaylar."""
    events = []
    opened = pd.Timestamp(pos["opened_at"])
    future = df[df["close_time"] > opened]
    if future.empty:
        return pos, events

    entry = pos["entry"]
    stop = pos["current_stop"]
    hit = set(pos.get("tps_hit", []))

    for _, c in future.iterrows():
        # KOTUMSER: ayni mumda ikisi de varsa once stop calisti say
        if c["low"] <= stop:
            if not hit:
                pos["realized_r"] = -1.0
                pos["status"] = "stopped"
                events.append("🛑 STOP oldu (-1.0R)")
            else:
                realized = sum(pos[f"r{i}"] for i in hit) / 3.0
                pos["realized_r"] = realized
                pos["status"] = "closed_be" if MOVE_STOP_TO_BE_AFTER_TP1 else "stopped"
                events.append(f"🔒 Kalan kisim "
                              f"{'BE' if MOVE_STOP_TO_BE_AFTER_TP1 else 'stop'}'ta kapandi "
                              f"(toplam {realized:+.2f}R)")
            pos["closed_at"] = c["close_time"].isoformat()
            pos["tps_hit"] = sorted(hit)
            pos["current_stop"] = stop
            return pos, events

        for i in (1, 2, 3):
            if i not in hit and c["high"] >= pos[f"tp{i}"]:
                hit.add(i)
                events.append(f"✅ TP{i} vuruldu ({pos[f'r{i}']:.1f}R, 1/3 pozisyon)")
                if i == 1 and MOVE_STOP_TO_BE_AFTER_TP1:
                    stop = entry
                    events.append("🔁 Stop giris seviyesine (BE) cekildi")

        if 3 in hit:
            pos["realized_r"] = sum(pos[f"r{i}"] for i in (1, 2, 3)) / 3.0
            pos["status"] = "target_done"
            pos["closed_at"] = c["close_time"].isoformat()
            pos["tps_hit"] = sorted(hit)
            pos["current_stop"] = stop
            events.append(f"🏁 Tum hedefler tamamlandi ({pos['realized_r']:+.2f}R)")
            return pos, events

    pos["tps_hit"] = sorted(hit)
    pos["current_stop"] = stop
    pos["unrealized_r"] = (float(future["close"].iloc[-1]) - entry) / pos["risk"]

    age_days = (datetime.now(timezone.utc) - opened.to_pydatetime()).total_seconds() / 86400
    if age_days > POSITION_MAX_DAYS and pos["status"] == "open":
        realized = sum(pos[f"r{i}"] for i in hit) / 3.0 if hit else pos["unrealized_r"] / 3.0
        pos["realized_r"] = realized
        pos["status"] = "timeout"
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        events.append(f"⏳ {POSITION_MAX_DAYS} gun doldu, kapatildi ({realized:+.2f}R)")

    return pos, events


def build_summary(positions):
    closed = [p for p in positions if p["status"] != "open"]
    open_ps = [p for p in positions if p["status"] == "open"]

    if not closed:
        return (f"📊 <b>OZET</b>\n\nHenuz kapanmis pozisyon yok.\n"
                f"Acik pozisyon: {len(open_ps)}")

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

    lines.append("\n<b>EMA bazinda</b>")
    for ema in EMA_PERIODS:
        g = [p for p in closed if p["ema_period"] == ema]
        if g:
            r = sum(p.get("realized_r", 0) for p in g)
            w = len([p for p in g if p.get("realized_r", 0) > 0])
            lines.append(f"  EMA{ema}: {len(g)} islem | {r:+.2f}R | basari %{w/len(g)*100:.0f}")

    lines.append("\n<b>Zaman dilimi bazinda</b>")
    for iv in INTERVALS:
        g = [p for p in closed if p["interval"] == iv]
        if g:
            r = sum(p.get("realized_r", 0) for p in g)
            w = len([p for p in g if p.get("realized_r", 0) > 0])
            lines.append(f"  {iv}: {len(g)} islem | {r:+.2f}R | basari %{w/len(g)*100:.0f}")

    lines.append("\n<b>Kapanis turu</b>")
    for st, lbl in [("stopped", "🛑 Stop"), ("closed_be", "🔒 BE/kismi"),
                    ("target_done", "🏁 Tum TP"), ("timeout", "⏳ Zaman asimi")]:
        g = [p for p in closed if p["status"] == st]
        if g:
            lines.append(f"  {lbl}: {len(g)} islem | {sum(p.get('realized_r',0) for p in g):+.2f}R")

    best = max(closed, key=lambda p: p.get("realized_r", 0))
    worst = min(closed, key=lambda p: p.get("realized_r", 0))
    lines.append(f"\n🥇 En iyi: {best['symbol']} {best['interval']} {best.get('realized_r',0):+.2f}R")
    lines.append(f"🥶 En kotu: {worst['symbol']} {worst['interval']} {worst.get('realized_r',0):+.2f}R")

    if open_ps:
        lines.append("\n<b>Acik pozisyonlar</b>")
        for p in sorted(open_ps, key=lambda x: x.get("unrealized_r", 0), reverse=True)[:10]:
            tps = "".join(f"✅{i}" for i in p.get("tps_hit", [])) or "—"
            lines.append(f"  {p['symbol']} {p['interval']} EMA{p['ema_period']}: "
                         f"{p.get('unrealized_r',0):+.2f}R {tps}")

    lines.append("\n<i>Varsayimsal: her TP'de 1/3 kapanis, TP1 sonrasi stop BE. "
                 "Komisyon/kayma dahil degil.</i>")
    return "\n".join(lines)


# ==================== GRAFIK ====================
def make_chart(df, symbol, interval, ema_period, plan):
    os.makedirs(CHART_DIR, exist_ok=True)
    d = df.tail(CHART_CANDLES).copy()
    d = d.set_index(pd.DatetimeIndex(d["close_time"]))
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                          "close": "Close", "volume": "Volume"})
    aps = [mpf.make_addplot(d[f"ema{p}"], color={55: "orange", 99: "purple"}[p], width=1.1)
           for p in EMA_PERIODS]
    hl = dict(hlines=[plan["stop"], plan["entry"], plan["tp1"], plan["tp2"], plan["tp3"]],
              colors=["red", "white", "#90ee90", "#2ecc71", "#f1c40f"],
              linestyle="--", linewidths=1.0)
    path = os.path.join(CHART_DIR, f"{symbol}_{interval}.png")
    mpf.plot(d, type="candle", style="binance", addplot=aps, volume=True,
             title=f"{symbol} - {interval}  (EMA{ema_period} temasi)",
             hlines=hl, savefig=dict(fname=path, dpi=130, bbox_inches="tight"))
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
                return json.load(f)
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

    # ---- 1) ACIK POZISYONLARI GUNCELLE ----
    open_ps = [p for p in positions if p["status"] == "open"]
    print(f"{len(open_ps)} acik pozisyon kontrol ediliyor...")
    for pos in open_ps:
        df = get_klines(pos["symbol"], pos["interval"])
        if df is None:
            continue
        _, events = evaluate_position(pos, df)
        if events:
            head = (f"📌 <b>{pos['symbol']}</b> {pos['interval']} EMA{pos['ema_period']}\n"
                    f"Giris: {pos['entry']:.6g} | Acilis: {pos['opened_at'][:10]}")
            tg_send(head + "\n" + "\n".join(events), TOPIC_RESULTS)
        time.sleep(0.05)

    # ---- 2) YENI SINYALLERI TARA ----
    symbols = get_usdt_symbols()
    print(f"{len(symbols)} sembol taranacak...")
    new_count = 0

    for symbol in symbols:
        try:
            if any(p["symbol"] == symbol and p["status"] == "open" for p in positions):
                continue  # zaten acik pozisyon var
            recent = [p for p in positions if p["symbol"] == symbol]
            if recent:
                last = max(pd.Timestamp(p["opened_at"]) for p in recent)
                if (now - last.to_pydatetime()).total_seconds() / 3600 < DEDUP_COOLDOWN_HOURS:
                    continue

            dfs, rallies = {}, {}
            for iv in INTERVALS:
                df = get_klines(symbol, iv)
                dfs[iv] = add_emas(df) if df is not None else None
                if dfs[iv] is not None:
                    rallies[iv] = find_rally(dfs[iv])
                time.sleep(0.04)

            trigger = None
            for iv, (ok, rpct, hi, lo, pb, days) in rallies.items():
                if not ok or pb < PULLBACK_MIN_PCT:
                    continue
                last = dfs[iv].iloc[-1]
                for p in EMA_PERIODS:
                    if abs(last["close"] - last[f"ema{p}"]) / last[f"ema{p}"] <= TOUCH_TOLERANCE_PCT:
                        plan = compute_trade_plan(lo, hi, float(last["close"]))
                        if plan:
                            trigger = (iv, p, rpct, hi, lo, pb, days, plan)
                            break
                if trigger:
                    break

            if not trigger:
                continue

            iv, ema_p, rpct, hi, lo, pb, days, plan = trigger

            msg = (f"🔔 <b>{symbol}</b>  [{iv} / EMA{ema_p}]\n"
                   f"Yukselis: %{rpct*100:.1f} ({days:.1f} gunde)\n"
                   f"Zirve {hi:.6g} → simdi {plan['entry']:.6g} (%{pb*100:.1f} geri cekildi)\n\n"
                   + format_ema_lines(ema_distances(dfs))
                   + "\n\n📋 <b>Islem Plani</b>\n" + format_plan(plan))

            sent = False
            try:
                chart = make_chart(dfs[iv], symbol, iv, ema_p, plan)
                sent = tg_photo(chart, msg, TOPIC_SIGNALS)
            except Exception as e:
                print(f"{symbol} grafik hatasi: {e}")
            if not sent:
                tg_send(msg, TOPIC_SIGNALS)

            positions.append({
                "symbol": symbol, "interval": iv, "ema_period": ema_p,
                "opened_at": now.isoformat(), "status": "open",
                "entry": plan["entry"], "stop": plan["stop"], "current_stop": plan["stop"],
                "risk": plan["risk"],
                "tp1": plan["tp1"], "tp2": plan["tp2"], "tp3": plan["tp3"],
                "r1": plan["r1"], "r2": plan["r2"], "r3": plan["r3"],
                "rally_pct": rpct, "rally_days": days,
                "tps_hit": [], "realized_r": 0.0, "unrealized_r": 0.0,
            })
            new_count += 1

        except Exception as e:
            print(f"{symbol} hata: {e}")
            continue

    print(f"{new_count} yeni sinyal.")

    # ---- 3) OZET ----
    if SUMMARY_EVERY_RUN or now.hour < 4:
        tg_send(build_summary(positions), TOPIC_SUMMARY)

    save_positions(positions)
    print("Tamamlandi.")


if __name__ == "__main__":
    main()
