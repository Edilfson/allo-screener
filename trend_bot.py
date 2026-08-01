"""
TREND TAKIBI BOTU  (MA50)
==========================
Tek kural: BTC gunluk kapanisi 50 gunluk ortalamanin
  USTUNDE ise -> YATIRIMDA KAL
  ALTINDA ise -> NAKITTE KAL

Gunde bir kez calisir, durum DEGISTIGINDE alarm verir.
Yilda ortalama 6-10 sinyal beklenir.

Backtest (Ags 2022 - Ags 2026, 4 yil):
  BTC al-tut     : +176%  | max dusus -53%
  BTC trend MA50 : +273%  | max dusus -26%   <- bu sistem
Kar tek tek islemlerden degil, BUYUK DUSUSLERDEN KACMAKTAN gelir.

UYARI: Yatirim tavsiyesi degildir. Gecmis performans gelecegi garanti etmez.
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
import matplotlib.pyplot as plt

BASE = "https://data-api.binance.vision"
TOP_N = int(os.environ.get("TREND_TOP", "50"))     # hacme gore ilk N coin
SYMBOL = os.environ.get("TREND_SYMBOL", "")        # dolu ise sadece o coin
MA_DAYS = int(os.environ.get("TREND_MA", "50"))
STATE_FILE = "trend_state.json"

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")
TOPIC = os.environ.get("TOPIC_SIGNALS")


def top_symbols(n):
    r = requests.get(f"{BASE}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    rows = []
    for t in r.json():
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        b = sym[:-4]
        if any(b.endswith(x) for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        if "USD" in b and len(b) <= 6:      # stablecoinler
            continue
        rows.append((sym, float(t.get("quoteVolume", 0))))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:n]]


def klines(symbol, limit=300):
    r = requests.get(f"{BASE}/api/v3/klines",
                     params={"symbol": symbol, "interval": "1d", "limit": limit},
                     timeout=30)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=["ts", "o", "h", "l", "c", "v",
                                    "ct", "qav", "n", "tb", "tq", "ig"])
    for x in ("o", "h", "l", "c"):
        df[x] = df[x].astype(float)
    df["dt"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
    # KAPANMAMIS son mumu at (karar sadece kapali mumla verilir)
    df = df[df["dt"] <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)
    return df


def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return False
    d = {"chat_id": TG_CHAT, "text": text[:4090], "parse_mode": "HTML"}
    if TOPIC:
        d["message_thread_id"] = TOPIC
    try:
        return requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                             data=d, timeout=20).status_code == 200
    except Exception as e:
        print("TG hata:", e)
        return False


def tg_photo(path, caption):
    if not TG_TOKEN or not TG_CHAT:
        print("[foto]", path, caption[:200])
        return False
    d = {"chat_id": TG_CHAT, "caption": caption[:1000], "parse_mode": "HTML"}
    if TOPIC:
        d["message_thread_id"] = TOPIC
    try:
        with open(path, "rb") as f:
            return requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                                 data=d, files={"photo": f}, timeout=45).status_code == 200
    except Exception as e:
        print("TG foto hata:", e)
        return False

def cizim(df, ma, durum, sym):
    """Fiyat + MA + al/sat noktalari."""
    d = df.tail(240).copy()
    m = ma.tail(240)
    poz = (d["c"].values > m.values)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#161a25")
    ax.set_facecolor("#161a25")
    ax.plot(d["dt"], d["c"], color="#eaecef", lw=1.4, label="BTC")
    ax.plot(d["dt"], m, color="#f0b90b", lw=1.6, label=f"MA{MA_DAYS}")
    ax.fill_between(d["dt"], d["c"].min() * 0.97, d["c"].max() * 1.03,
                    where=poz, color="#26a69a", alpha=0.10)
    gec = np.where(poz[1:] != poz[:-1])[0] + 1
    for i in gec[-12:]:
        ax.scatter(d["dt"].iloc[i], d["c"].iloc[i], s=70, zorder=5,
                   color="#26a69a" if poz[i] else "#ef5350",
                   marker="^" if poz[i] else "v")
    ax.set_title(f"{sym}  gunluk  |  MA{MA_DAYS} trend takibi  |  DURUM: {durum}",
                 color="#eaecef", fontsize=13, pad=12)
    ax.tick_params(colors="#b7bdc6", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2b3139")
    ax.grid(color="#2b3139", lw=0.5, alpha=0.6)
    ax.legend(facecolor="#161a25", edgecolor="#2b3139", labelcolor="#b7bdc6", fontsize=9)
    os.makedirs("charts", exist_ok=True)
    path = f"charts/trend_{sym}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#161a25")
    plt.close(fig)
    return path


def tek_coin(sym, durumlar):
    """Bir coin icin durum. Return: (satir, degisti, yatirimda)"""
    df = klines(sym)
    if df is None or len(df) < MA_DAYS + 5:
        return None, None, None
    ma = df["c"].rolling(MA_DAYS).mean()
    fiyat = float(df["c"].iloc[-1])
    ma_now = float(ma.iloc[-1])
    if np.isnan(ma_now):
        return None, None, None
    yatirimda = fiyat > ma_now
    fark = (fiyat - ma_now) / ma_now * 100
    durum = "YATIRIMDA" if yatirimda else "NAKITTE"

    onceki = durumlar.get(sym)
    degisti = (onceki is not None) and (onceki != yatirimda)

    isaret = "\U0001F7E2" if yatirimda else "\U0001F534"
    satir = (f"{isaret} {sym[:-4]:<8} {fiyat:>12,.4f} | "
             f"MA{MA_DAYS} {ma_now:>12,.4f} ({fark:+6.1f}%)")

    if degisti:
        if yatirimda:
            mesaj = (f"\U0001F7E2 <b>AL SINYALI - {sym[:-4]}</b>\n"
                     f"Fiyat {MA_DAYS} gunluk ortalamanin USTUNE cikti.\n\n"
                     f"Fiyat: {fiyat:,.4f}\nMA{MA_DAYS}: {ma_now:,.4f}  ({fark:+.1f}%)\n"
                     f"Tarih: {df['dt'].iloc[-1].date()}\n\n"
                     f"Yapilacak: pozisyona gir / yatirimda kal.")
        else:
            mesaj = (f"\U0001F534 <b>SAT SINYALI - {sym[:-4]}</b>\n"
                     f"Fiyat {MA_DAYS} gunluk ortalamanin ALTINA indi.\n\n"
                     f"Fiyat: {fiyat:,.4f}\nMA{MA_DAYS}: {ma_now:,.4f}  ({fark:+.1f}%)\n"
                     f"Tarih: {df['dt'].iloc[-1].date()}\n\n"
                     f"Yapilacak: pozisyondan cik, nakitte bekle.")
        try:
            p = cizim(df, ma, durum, sym)
            if not tg_photo(p, mesaj):
                tg_send(mesaj)
        except Exception as e:
            print(f"{sym} grafik hatasi: {e}")
            tg_send(mesaj)
    return satir, degisti, yatirimda


def main():
    durumlar = {}
    if os.path.exists(STATE_FILE):
        try:
            durumlar = json.load(open(STATE_FILE)).get("durumlar", {})
        except Exception:
            durumlar = {}

    semboller = [SYMBOL] if SYMBOL else top_symbols(TOP_N)
    print(f"{len(semboller)} coin kontrol ediliyor (MA{MA_DAYS})...")

    satirlar, degisim, yeni_durum = [], 0, {}
    for i, sym in enumerate(semboller, 1):
        try:
            satir, degisti, yat = tek_coin(sym, durumlar)
            if satir is None:
                continue
            satirlar.append(satir)
            yeni_durum[sym] = bool(yat)
            if degisti:
                degisim += 1
        except Exception as e:
            print(f"{sym} hata: {e}")
        if i % 10 == 0:
            print(f"  {i}/{len(semboller)}")
        time.sleep(0.05)

    yat = [s for s in satirlar if s.startswith("\U0001F7E2")]
    nak = [s for s in satirlar if s.startswith("\U0001F534")]
    ek = f" | {degisim} degisim" if degisim else ""
    ozet = (f"\U0001F4CA <b>TREND DURUMU (MA{MA_DAYS})</b>\n"
            f"{len(yat)} coin YATIRIMDA | {len(nak)} coin NAKITTE{ek}\n\n"
            f"<code>" + chr(10).join(satirlar[:40]) + "</code>")
    if len(satirlar) > 40:
        ozet += f"\n<i>... ve {len(satirlar)-40} coin daha</i>"
    tg_send(ozet)

    json.dump({"durumlar": yeni_durum,
               "guncelleme": datetime.now(timezone.utc).isoformat()},
              open(STATE_FILE, "w"), indent=1)
    print(f"Bitti. {len(yat)} yatirimda, {len(nak)} nakitte, {degisim} degisim.")


if __name__ == "__main__":
    main()
