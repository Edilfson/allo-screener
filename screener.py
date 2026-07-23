"""
Binance Futures (USDT-M Perp) EMA Pullback Screener
-----------------------------------------------------
Mantik:
1) Her coin icin son N muma bakip, bir "swing low -> swing high" hareketi ariyoruz.
2) Bu hareket >= RALLY_MIN_PCT (%50) ise, coin "guclu yukselis yapmis" sayilir.
3) Fiyat zirveden geri cekilmisse (PULLBACK_MIN_PCT kadar) ve simdi EMA55 veya EMA99'a
   TOUCH_TOLERANCE_PCT kadar yakinsa/temas etmisse -> ALARM.
4) Ayni sinyal icin tekrar tekrar alarm atmamak icin bir "state" dosyasi (JSON) tutulur.
   GitHub Actions bu dosyayi her calistirmadan sonra repoya geri commit eder.

Not: Bu kesin bir "sinyal her zaman dogru" garantisi degildir; sadece taramayi
otomatiklestirir. Nihai karari sen verirsin.
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

# ---------------- AYARLAR ----------------
INTERVALS = ["4h", "1d"]          # taranacak zaman dilimleri
EMA_PERIODS = [55, 99]            # 99 ~ 100 gunluk/mumluk ortalama muadili
RALLY_MIN_PCT = 0.50              # swing low -> swing high min %50 yukselis
RALLY_MAX_DAYS = 30               # yukselis (swing low -> swing high) en fazla kac GUN icinde olmus olmali
PULLBACK_MIN_PCT = 0.05           # zirveden en az %5 geri cekilmis olmali
TOUCH_TOLERANCE_PCT = 0.015       # EMA'ya %1.5 mesafe = "temas/yaklasti" sayilir
LOOKBACK_CANDLES = 250            # her seri icin cekilecek mum sayisi
DEDUP_COOLDOWN_HOURS = 20         # ayni sinyal icin tekrar alarm atmadan once bekle
CHART_CANDLES = 120               # grafikte gosterilecek son mum sayisi

STATE_FILE = "screener_state.json"
CHART_DIR = "charts"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BASE_URL = "https://fapi.binance.com"


def get_perp_usdt_symbols():
    """Binance Futures'taki tum USDT-M perpetual sembolleri getirir."""
    r = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=15)
    r.raise_for_status()
    data = r.json()
    symbols = []
    for s in data["symbols"]:
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            symbols.append(s["symbol"])
    return sorted(symbols)


def get_klines(symbol, interval, limit=LOOKBACK_CANDLES):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=15)
    if r.status_code != 200:
        return None
    raw = r.json()
    if not raw or len(raw) < 60:
        return None
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def add_emas(df):
    for p in EMA_PERIODS:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def find_rally_and_check(df):
    """
    Son RALLY_MAX_DAYS gun icindeki mumlarda: en dusuk kapanistan sonra gelen en
    yuksek kapanisi bul, yukselis >= RALLY_MIN_PCT mi VE bu yukselis RALLY_MAX_DAYS
    gun icinde mi olmus kontrol et. Sonra son fiyatin zirveden PULLBACK_MIN_PCT
    kadar geri cekilip cekilmedigine bak.
    Return: (rally_ok, rally_pct, swing_high, pullback_pct, rally_days)
    """
    now_ts = df["close_time"].iloc[-1]
    window_start = now_ts - pd.Timedelta(days=RALLY_MAX_DAYS)
    window_df = df[df["close_time"] >= window_start].reset_index(drop=True)

    closes = window_df["close"].values
    times = window_df["close_time"].values
    n = len(closes)
    if n < 5:
        return False, 0, 0, 0, 0

    # pencere icindeki en dusuk kapanisi bul, ondan sonraki en yuksek kapanisi bul
    min_idx = int(np.argmin(closes))
    if min_idx >= n - 2:
        return False, 0, 0, 0, 0
    after_min = closes[min_idx:]
    max_idx_rel = int(np.argmax(after_min))
    max_idx = min_idx + max_idx_rel

    swing_low = closes[min_idx]
    swing_high = closes[max_idx]

    if swing_low <= 0:
        return False, 0, 0, 0, 0

    rally_pct = (swing_high - swing_low) / swing_low
    rally_days = (pd.Timestamp(times[max_idx]) - pd.Timestamp(times[min_idx])).total_seconds() / 86400

    if rally_pct < RALLY_MIN_PCT:
        return False, rally_pct, swing_high, 0, rally_days

    last_close = closes[-1]
    pullback_pct = (swing_high - last_close) / swing_high
    return True, rally_pct, swing_high, pullback_pct, rally_days


def check_ema_touch(df):
    """Son mumun EMA55 / EMA99'a yakinligini kontrol eder."""
    last = df.iloc[-1]
    touches = []
    for p in EMA_PERIODS:
        ema_val = last[f"ema{p}"]
        dist_pct = abs(last["close"] - ema_val) / ema_val
        if dist_pct <= TOUCH_TOLERANCE_PCT:
            touches.append((p, ema_val, dist_pct))
    return touches


def all_ema_distances(dfs):
    """
    dfs: {"4h": df_or_None, "1d": df_or_None}
    Her (interval, ema_period) kombinasyonu icin mesafe yuzdesini hesaplar.
    Return: liste of dict {interval, ema_period, ema_val, last_close, dist_pct}
    """
    rows = []
    for interval, df in dfs.items():
        if df is None:
            continue
        last = df.iloc[-1]
        for p in EMA_PERIODS:
            ema_val = last[f"ema{p}"]
            last_close = last["close"]
            dist_pct = abs(last_close - ema_val) / ema_val
            rows.append({
                "interval": interval,
                "ema_period": p,
                "ema_val": ema_val,
                "last_close": last_close,
                "dist_pct": dist_pct,
            })
    return rows


def format_ema_lines(rows):
    """En yakin noktaya 🎯, digerlerine 📍 verip satirlari olusturur."""
    if not rows:
        return ""
    closest_idx = min(range(len(rows)), key=lambda i: rows[i]["dist_pct"])
    lines = []
    for i, row in enumerate(rows):
        emoji = "🎯" if i == closest_idx else "📍"
        lines.append(
            f"{emoji} {row['interval']} EMA{row['ema_period']}: {row['ema_val']:.5f} "
            f"(mesafe %{row['dist_pct']*100:.2f})"
        )
    return "\n".join(lines)


def make_chart(df, symbol, interval, ema_touch_period=None):
    """Son CHART_CANDLES muma ait mum grafigi + EMA55/EMA99 cizgileriyle PNG olusturur."""
    os.makedirs(CHART_DIR, exist_ok=True)
    plot_df = df.tail(CHART_CANDLES).copy()
    plot_df = plot_df.set_index(pd.DatetimeIndex(plot_df["close_time"]))
    plot_df = plot_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })

    add_plots = []
    colors = {55: "orange", 99: "purple"}
    for p in EMA_PERIODS:
        add_plots.append(
            mpf.make_addplot(plot_df[f"ema{p}"], color=colors.get(p, "blue"), width=1.1)
        )

    title = f"{symbol} - {interval}"
    if ema_touch_period:
        title += f"  (EMA{ema_touch_period} temasi)"

    path = os.path.join(CHART_DIR, f"{symbol}_{interval}.png")
    mpf.plot(
        plot_df,
        type="candle",
        style="binance",
        addplot=add_plots,
        volume=True,
        title=title,
        savefig=dict(fname=path, dpi=130, bbox_inches="tight"),
    )
    return path


def send_telegram_photo(photo_path, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bilgisi eksik, foto gonderilemedi:", photo_path)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    # Telegram caption limiti 1024 karakter
    if len(caption) > 1000:
        caption = caption[:1000] + "\n... (devami sonraki mesajda)"
    try:
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
            r = requests.post(url, data=data, files=files, timeout=30)
        return r.status_code == 200
    except Exception as e:
        print("Telegram foto gonderim hatasi:", e)
        return False


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram bilgisi eksik, mesaj konsola yazdiriliyor:\n", msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("Telegram gonderim hatasi:", e)


def main():
    state = load_state()
    symbols = get_perp_usdt_symbols()
    print(f"{len(symbols)} sembol taranacak...")

    now = datetime.now(timezone.utc)
    alerts = []

    for symbol in symbols:
        try:
            dfs = {}
            rally_info = {}  # interval -> (rally_ok, rally_pct, swing_high, pullback_pct, rally_days)

            for interval in INTERVALS:
                df = get_klines(symbol, interval)
                if df is None:
                    dfs[interval] = None
                    continue
                df = add_emas(df)
                dfs[interval] = df
                rally_info[interval] = find_rally_and_check(df)
                time.sleep(0.05)  # rate limit icin kucuk bekleme

            # herhangi bir interval'da rally + pullback + EMA temasi var mi?
            triggered_intervals = []
            for interval, info in rally_info.items():
                rally_ok, rally_pct, swing_high, pullback_pct, rally_days = info
                if not rally_ok or pullback_pct < PULLBACK_MIN_PCT:
                    continue
                touches = check_ema_touch(dfs[interval])
                if touches:
                    triggered_intervals.append(interval)

            if not triggered_intervals:
                continue

            # dedup: bu sembol icin herhangi bir triggered interval yakin zamanda alarm verdiyse atla
            key = f"{symbol}_signal"
            last_alert = state.get(key)
            if last_alert:
                last_dt = datetime.fromisoformat(last_alert)
                hours_since = (now - last_dt).total_seconds() / 3600
                if hours_since < DEDUP_COOLDOWN_HOURS:
                    continue
            state[key] = now.isoformat()

            # mesaji olustur: tetikleyen interval(lar)in yukselis bilgisi + 4 EMA noktasinin tamami
            rally_lines = []
            for interval in triggered_intervals:
                rally_ok, rally_pct, swing_high, pullback_pct, rally_days = rally_info[interval]
                last_close = dfs[interval].iloc[-1]["close"]
                rally_lines.append(
                    f"[{interval}] Yukselis: %{rally_pct*100:.1f} ({rally_days:.1f} gunde) | "
                    f"Zirve: {swing_high:.5f} -> Simdi: {last_close:.5f} "
                    f"(%{pullback_pct*100:.1f} geri cekildi)"
                )

            ema_rows = all_ema_distances(dfs)
            ema_lines = format_ema_lines(ema_rows)

            msg = (
                f"🔔 <b>{symbol}</b>\n"
                + "\n".join(rally_lines)
                + "\n\n"
                + ema_lines
            )
            alerts.append(msg)

            # en yakin EMA noktasinin oldugu zaman diliminden grafik olustur
            closest_row = min(ema_rows, key=lambda r: r["dist_pct"]) if ema_rows else None
            if closest_row:
                try:
                    chart_path = make_chart(
                        dfs[closest_row["interval"]], symbol,
                        closest_row["interval"], closest_row["ema_period"]
                    )
                    ok = send_telegram_photo(chart_path, msg)
                    if not ok:
                        send_telegram(msg)
                except Exception as e:
                    print(f"{symbol} grafik hatasi: {e}")
                    send_telegram(msg)
            else:
                send_telegram(msg)

        except Exception as e:
            print(f"{symbol} hata: {e}")
            continue

    print(f"{len(alerts)} sinyal bulundu.")
    save_state(state)


if __name__ == "__main__":
    main()
