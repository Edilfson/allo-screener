"""
Komut Botu - Telegram'dan istek uzerine rapor
==============================================
Her calismada bot mesajlarini (getUpdates) okur, komutlari isler, cevaplar.
GitHub Actions cron'u ile ~10 dakikada bir calisir (anlik degil; en gec
10-15 dk icinde cevap gelir).

Komutlar:
  /report (veya /rapor) -> genel ozet (toplam R, basari, strateji A/B ...)
  /acik                 -> acik islemler + ANLIK fiyatla guncel R durumu
  /coin SOL             -> o coinin islemi: grafik (giris noktasi isaretli,
                           stop/TP cizgili) + durum ozeti
  /yardim               -> komut listesi

Guvenlik: sadece TELEGRAM_CHAT_ID'deki mesajlar islenir.
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = str(os.environ.get("TELEGRAM_CHAT_ID", ""))
BASE_URL = "https://data-api.binance.vision"
POS_FILE = "positions.json"
BOT_STATE = "bot_state.json"


def tg(method, **params):
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
                          data=params, timeout=20)
        return r.json()
    except Exception as e:
        print("TG hata:", e)
        return {}


def tg_photo(path, caption, thread=None):
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000], "parse_mode": "HTML"}
    if thread:
        data["message_thread_id"] = thread
    try:
        with open(path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          data=data, files={"photo": f}, timeout=45)
    except Exception as e:
        print("foto hata:", e)


def reply(text, thread=None):
    p = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:4090], "parse_mode": "HTML"}
    if thread:
        p["message_thread_id"] = thread
    tg("sendMessage", **p)


def load_positions():
    if os.path.exists(POS_FILE):
        try:
            with open(POS_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def get_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=10)
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception:
        pass
    return None


def unrealized(pos, price):
    done = sum(t["w"] * t["r"] for t in pos.get("tps", []) if t.get("hit"))
    rem = sum(t["w"] for t in pos.get("tps", []) if not t.get("hit"))
    return done + rem * (price - pos["entry"]) / pos["risk"]


def cmd_report(thread):
    import screener
    positions = load_positions()
    reply(screener.build_summary([screener.migrate_position(p) for p in positions]), thread)


def cmd_acik(thread):
    positions = [p for p in load_positions() if p.get("status") == "open"]
    if not positions:
        reply("Su an acik islem yok.", thread)
        return
    lines = [f"📂 <b>ACIK ISLEMLER</b> ({len(positions)}) — anlik fiyatla:\n"]
    for p in sorted(positions, key=lambda x: x["opened_at"], reverse=True):
        price = get_price(p["symbol"])
        if price:
            ur = unrealized(p, price)
            hits = "".join(f"✅{i}" for i, t in enumerate(p.get("tps", []), 1) if t.get("hit")) or "—"
            be = " (BE)" if p.get("current_stop", 0) >= p["entry"] else ""
            lines.append(f"{p['symbol']} {p['interval']} [{p.get('strategy','?')}]\n"
                         f"  Giris {p['entry']:.6g} → simdi {price:.6g} | "
                         f"<b>{ur:+.2f}R</b> | TP:{hits} | stop{be}: {p.get('current_stop',p['stop']):.6g}")
        else:
            lines.append(f"{p['symbol']}: fiyat alinamadi")
        time.sleep(0.05)
    reply("\n".join(lines), thread)


def cmd_coin(arg, thread):
    sym = arg.strip().upper().replace("/", "")
    if not sym:
        reply("Kullanim: /coin SOL  (veya /coin SOLUSDT)", thread)
        return
    if not sym.endswith("USDT"):
        sym += "USDT"
    positions = load_positions()
    mine = [p for p in positions if p["symbol"] == sym]
    if not mine:
        price = get_price(sym)
        extra = f"\nAnlik fiyat: {price:.6g}" if price else ""
        reply(f"{sym} icin kayitli islem yok.{extra}", thread)
        return
    open_ps = [p for p in mine if p["status"] == "open"]
    pos = open_ps[0] if open_ps else sorted(mine, key=lambda x: x["opened_at"])[-1]

    price = get_price(sym)
    durum = "ACIK" if pos["status"] == "open" else pos["status"]
    lines = [f"📌 <b>{sym}</b> {pos['interval']} [{pos.get('strategy','?')}] — {durum}",
             f"Giris: {pos['entry']:.6g} ({pos['opened_at'][:16].replace('T',' ')} UTC)"]
    if price and pos["status"] == "open":
        lines.append(f"Simdi: {price:.6g} → <b>{unrealized(pos, price):+.2f}R</b>")
    elif pos["status"] != "open":
        lines.append(f"Sonuc: {pos.get('realized_r',0):+.2f}R")
    be = " (BE'de)" if pos.get("current_stop", 0) >= pos["entry"] else ""
    lines.append(f"Stop: {pos.get('current_stop', pos['stop']):.6g}{be}")
    for i, t in enumerate(pos.get("tps", []), 1):
        ok = "✅" if t.get("hit") else "⬜"
        lines.append(f"{ok} TP{i} (%{t['w']*100:.0f}): {t['p']:.6g} → {t['r']:.1f}R")
    caption = "\n".join(lines)

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import mplfinance as mpf
        import screener as S

        df = S.get_klines(sym, pos["interval"])
        if df is None:
            reply(caption + "\n(grafik icin veri alinamadi)", thread)
            return
        df = S.add_emas(df)
        d = df.tail(150).copy()
        d = d.set_index(pd.DatetimeIndex(d["close_time"]))
        d = d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "volume": "Volume"})
        aps = [mpf.make_addplot(d[f"ema{p}"], color={55: "orange", 99: "purple"}[p], width=1.1)
               for p in (55, 99)]
        prices = [pos.get("current_stop", pos["stop"]), pos["entry"]] + \
                 [t["p"] for t in pos.get("tps", [])]
        colors = ["red", "white", "#90ee90", "#2ecc71", "#f1c40f"][:len(prices)]
        opened = pd.Timestamp(pos["opened_at"])
        vl = {}
        if d.index[0] <= opened <= d.index[-1]:
            vl = dict(vlines=dict(vlines=[opened], colors=["#00e5ff"],
                                  linestyle=":", linewidths=1.4))
        os.makedirs("charts", exist_ok=True)
        path = f"charts/{sym}_cmd.png"
        mpf.plot(d, type="candle", style="binance", addplot=aps, volume=True,
                 title=f"{sym} - {pos['interval']}  [{pos.get('strategy','?').upper()}]",
                 hlines=dict(hlines=prices, colors=colors, linestyle="--", linewidths=1.0),
                 savefig=dict(fname=path, dpi=130, bbox_inches="tight"), **vl)
        tg_photo(path, caption, thread)
    except Exception as e:
        print("grafik hata:", e)
        reply(caption, thread)


def cmd_yardim(thread):
    reply("🤖 <b>Komutlar</b>\n"
          "/report — genel ozet (toplam R, basari, strateji A/B)\n"
          "/acik — acik islemler, anlik fiyatla guncel R\n"
          "/coin SOL — o coinin islemi, grafikle (giris noktasi isaretli)\n"
          "/yardim — bu liste\n\n"
          "<i>Not: bot ~10 dakikada bir kontrol eder; cevap en gec 10-15 dk icinde gelir.</i>",
          thread)


def main():
    state = {}
    if os.path.exists(BOT_STATE):
        try:
            with open(BOT_STATE) as f:
                state = json.load(f)
        except Exception:
            state = {}
    offset = state.get("offset", 0)

    res = tg("getUpdates", offset=offset + 1, timeout=0, allowed_updates='["message"]')
    updates = res.get("result", [])
    print(f"{len(updates)} guncelleme")

    for u in updates:
        offset = max(offset, u["update_id"])
        msg = u.get("message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = (msg.get("text") or "").strip()
        thread = msg.get("message_thread_id")
        if chat != TELEGRAM_CHAT_ID or not text.startswith("/"):
            continue
        parts = text.split(maxsplit=1)
        cmd = parts[0].split("@")[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        print("komut:", cmd, arg)
        try:
            if cmd in ("/report", "/rapor"):
                cmd_report(thread)
            elif cmd in ("/acik", "/open"):
                cmd_acik(thread)
            elif cmd == "/coin":
                cmd_coin(arg, thread)
            elif cmd in ("/yardim", "/help", "/start"):
                cmd_yardim(thread)
        except Exception as e:
            print("komut hatasi:", e)
            reply(f"Komut islenirken hata olustu: {e}", thread)

    with open(BOT_STATE, "w") as f:
        json.dump({"offset": offset,
                   "last_run": datetime.now(timezone.utc).isoformat()}, f)
    print("Tamamlandi. offset:", offset)


if __name__ == "__main__":
    main()
