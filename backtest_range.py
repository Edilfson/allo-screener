#!/usr/bin/env python3
"""
LOWER HIGH + RANGE USTU + LIKIDITE HEDEFI  --  SHORT/LONG KURULUMU BACKTEST
===========================================================================
Spesifikasyon: kurulum_spesifikasyonu.md (R1-R7)
Canli karsiligi: range_setup.py (screener icinde range_lh / range_hl)

KULLANIM
    python backtest_range.py --exchange binance_vision --symbol BTCUSDT --tf 4h --days 730 --oos

NOT: --sweep parametre optimizasyonu yapar. Cikan en iyi sonuc GERCEK
beklentiniz DEGILDIR (overfit). Optimizasyondan sonra mutlaka --oos ile
veriyi ikiye bolup ikinci yarida dogrulayin.
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1) VERI
# ----------------------------------------------------------------------

BINANCE_VISION = "https://data-api.binance.vision"


def fetch_binance(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Binance'in kisitlamasiz veri aynasindan mum ceker (ccxt gerekmez).
    GitHub Actions sunucularinda fapi 451 verdigi icin bu ayna kullanilir."""
    import requests
    tf_ms = {"1h": 3600, "2h": 7200, "4h": 14400, "12h": 43200,
             "1d": 86400}[timeframe] * 1000
    since = int(pd.Timestamp.utcnow().timestamp() * 1000) - days * 86_400_000
    rows = []
    while True:
        r = requests.get(f"{BINANCE_VISION}/api/v3/klines",
                         params={"symbol": symbol, "interval": timeframe,
                                 "startTime": since, "limit": 1000}, timeout=30)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        rows.extend([b[:6] for b in batch])
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("timestamp", "ts", "time", "date", "datetime") if c in df.columns), None)
    if tcol is None:
        sys.exit("CSV'de zaman kolonu bulunamadi.")
    col = df[tcol]
    if pd.api.types.is_numeric_dtype(col):
        unit = "ms" if float(col.iloc[0]) > 1e11 else "s"
        df["dt"] = pd.to_datetime(col, unit=unit, utc=True)
    else:
        df["dt"] = pd.to_datetime(col, utc=True, format="mixed")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df.sort_values("dt").reset_index(drop=True)


# ----------------------------------------------------------------------
# 2) SWING NOKTALARI (fraktal) - k bar SONRA teyit edilir (ileriye bakma yok)
# ----------------------------------------------------------------------

def swing_flags(high: np.ndarray, low: np.ndarray, k: int):
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        win_h = high[i - k:i + k + 1]
        if high[i] == win_h.max() and win_h.argmax() == k:
            sh[i] = True
        win_l = low[i - k:i + k + 1]
        if low[i] == win_l.min() and win_l.argmin() == k:
            sl[i] = True
    return sh, sl


DEFAULTS = dict(
    swing_k=3,
    htf_mult=1,
    side="both",
    max_conf_cost=1.0,
    lookback=120,
    min_range_pct=0.03,
    max_range_pct=0.15,
    trend_max=0.06,
    upper_zone=0.40,
    stop_buf=0.002,
    target_buf=0.002,
    min_rr=3.0,
    max_hold=200,
    fee_pct=0.0005,
    funding_8h=0.0001,
    risk_frac=0.01,
)


def build_htf(df: pd.DataFrame, m: int):
    """LTF barlari m'lik bloklara toplar."""
    n = len(df)
    nb = n // m
    hi = df["high"].values[:nb * m].reshape(nb, m).max(axis=1)
    lo = df["low"].values[:nb * m].reshape(nb, m).min(axis=1)
    cl = df["close"].values[:nb * m].reshape(nb, m)[:, -1]
    htf_end = np.arange(nb) * m + (m - 1)
    return hi, lo, cl, htf_end


def run_backtest(df: pd.DataFrame, tf_hours: float, p: dict):
    """R1-R7 kural setini uygular. side: -1 SHORT (lower high), +1 LONG (higher low)."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    m = max(1, int(p["htf_mult"]))

    # yapi (range + swingler) HTF'te, giris LTF'te
    H_hi, H_lo, H_cl, htf_end = build_htf(df, m)
    sh, sl = swing_flags(H_hi, H_lo, p["swing_k"])
    sh_blk, sl_blk = np.flatnonzero(sh), np.flatnonzero(sl)

    sides = []
    if p["side"] in ("short", "both"):
        sides.append(-1)
    if p["side"] in ("long", "both"):
        sides.append(+1)

    rej = dict(range=0, trend=0, no_swing=0, not_struct=0, not_zone=0,
               no_confirm=0, bad_levels=0, conf_cost=0, rr=0, PASS=0)

    trades = []
    busy_until = -1
    lb = p["lookback"]

    start = (lb + p["swing_k"] + 2) * m
    for i in range(start, n - 1):
        if i <= busy_until:
            continue

        b_last = (i // m) - 1
        if b_last < lb + p["swing_k"] + 1:
            continue

        # --- R1 range ---
        range_high = H_hi[b_last - lb:b_last + 1].max()
        range_low = H_lo[b_last - lb:b_last + 1].min()
        rng = range_high - range_low
        if rng <= 0:
            continue
        if not (p["min_range_pct"] <= rng / range_low <= p["max_range_pct"]):
            rej["range"] += 1
            continue

        # --- R2 trend filtresi ---
        drift = abs(H_cl[b_last] - H_cl[b_last - lb]) / H_cl[b_last - lb]
        if drift > p["trend_max"]:
            rej["trend"] += 1
            continue

        for side in sides:
            if i <= busy_until:
                break

            # --- R3 yapi: lower high (short) / higher low (long) ---
            blks = sh_blk if side == -1 else sl_blk
            px = H_hi if side == -1 else H_lo
            conf = blks[blks <= b_last - p["swing_k"]]      # ILERIYE BAKMA KORUMASI
            if len(conf) < 2:
                rej["no_swing"] += 1
                continue
            b_s, b_prev = conf[-1], conf[-2]
            if b_s < b_last - lb:
                rej["no_swing"] += 1
                continue

            if side * (px[b_s] - px[b_prev]) <= 0:
                rej["not_struct"] += 1
                continue

            if side == -1:
                in_zone = px[b_s] >= range_high - p["upper_zone"] * rng
            else:
                in_zone = px[b_s] <= range_low + p["upper_zone"] * rng
            if not in_zone:
                rej["not_zone"] += 1
                continue

            # --- R4 TEYIT ---
            trig = H_lo[b_s] if side == -1 else H_hi[b_s]
            if side == -1:
                broke_now, broke_prev = close[i] < trig, close[i - 1] < trig
            else:
                broke_now, broke_prev = close[i] > trig, close[i - 1] > trig
            if not (broke_now and not broke_prev):
                rej["no_confirm"] += 1
                continue

            # --- R5 seviyeler ---
            entry = close[i]
            if side == -1:
                s_px = max(H_hi[b_prev], H_hi[b_s])
                stop = s_px * (1 + p["stop_buf"])
                target = range_low * (1 + p["target_buf"])
            else:
                s_px = min(H_lo[b_prev], H_lo[b_s])
                stop = s_px * (1 - p["stop_buf"])
                target = range_high * (1 - p["target_buf"])

            risk = side * (entry - stop)
            reward = side * (target - entry)
            if risk <= 0 or reward <= 0:
                rej["bad_levels"] += 1
                continue

            # --- R7 teyit maliyeti ---
            struct_h = abs(s_px - target)
            conf_cost = abs(s_px - entry) / struct_h if struct_h > 0 else 9.9
            if conf_cost > p["max_conf_cost"]:
                rej["conf_cost"] += 1
                continue

            # --- R6 R:R ---
            rr = reward / risk
            if rr < p["min_rr"]:
                rej["rr"] += 1
                continue
            rej["PASS"] += 1

            # --- simulasyon (ayni bar stop+hedef -> KAYIP) ---
            outcome, bars_held, exit_px = "open", 0, None
            for j in range(i + 1, min(i + 1 + p["max_hold"], n)):
                bars_held = j - i
                hit_stop = high[j] >= stop if side == -1 else low[j] <= stop
                if hit_stop:
                    outcome, exit_px = "loss", stop
                    break
                hit_tgt = low[j] <= target if side == -1 else high[j] >= target
                if hit_tgt:
                    outcome, exit_px = "win", target
                    break
            if outcome == "open":
                j = min(i + p["max_hold"], n - 1)
                bars_held = j - i
                outcome, exit_px = "timeout", close[j]

            gross_R = side * (exit_px - entry) / risk
            stop_dist_pct = risk / entry
            fee_R = 2 * p["fee_pct"] / stop_dist_pct
            hours = bars_held * tf_hours
            funding_R = p["funding_8h"] * (hours / 8.0) / stop_dist_pct
            net_R = gross_R - fee_R - funding_R

            trades.append(dict(
                dt=df["dt"].iloc[i], side="SHORT" if side == -1 else "LONG",
                entry=entry, stop=stop, target=target,
                rr=rr, conf_cost=conf_cost, outcome=outcome, bars=bars_held,
                gross_R=gross_R, cost_R=fee_R + funding_R, net_R=net_R,
            ))
            busy_until = i + bars_held

    return pd.DataFrame(trades), rej


# ----------------------------------------------------------------------
# 4) RAPOR
# ----------------------------------------------------------------------

def print_funnel(rej):
    if not rej:
        return
    order = [("range", "Range filtresi"), ("trend", "Trend filtresi"),
             ("no_swing", "Swing yok"), ("not_struct", "Yapi yok (LH/HL)"),
             ("not_zone", "Range bolgesinde degil"), ("no_confirm", "Teyit yok"),
             ("bad_levels", "Gecersiz seviye"), ("conf_cost", "Teyit maliyeti yuksek (R7)"),
             ("rr", "R:R yetersiz (R6)"), ("PASS", ">>> ISLEME GIRDI")]
    print("\n  ELEME HUNISI (aday bar basina)")
    print("  " + "-" * 44)
    for k, name in order:
        if k in rej:
            print(f"  {name:<30} {rej[k]:>8,}")


def report(tr: pd.DataFrame, p: dict, label="", rej=None):
    if tr.empty:
        print(f"\n{label}  ->  HIC ISLEM URETILMEDI.")
        print_funnel(rej)
        return None

    n = len(tr)
    wins = int((tr.net_R > 0).sum())
    wr = wins / n
    exp_net = tr.net_R.mean()
    gains = tr.loc[tr.net_R > 0, "net_R"].sum()
    losses = -tr.loc[tr.net_R <= 0, "net_R"].sum()
    pf = gains / losses if losses > 0 else float("inf")

    eq = [1.0]
    for r in tr.net_R:
        eq.append(eq[-1] * (1 + p["risk_frac"] * r))
    eq = np.array(eq)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max() * 100)

    print(f"\n{'=' * 62}\n  {label}\n{'=' * 62}")
    print(f"  Islem sayisi        : {n}")
    print(f"  Kazanma orani       : %{wr*100:.1f}  ({wins}K / {n-wins}Z)")
    print(f"  Ort. R:R            : {tr.rr.mean():.2f}")
    print(f"  Beklenti (brut)     : {tr.gross_R.mean():+.3f}R")
    print(f"  Beklenti (net)      : {exp_net:+.3f}R   <- maliyet dusulmus")
    print(f"  Profit factor       : {pf:.2f}")
    print(f"  Toplam net          : {tr.net_R.sum():+.1f}R")
    print(f"  Max dusus (%{p['risk_frac']*100:.0f} risk) : %{dd:.1f}")
    print(f"  Ort. tutma          : {tr.bars.mean():.0f} bar")
    to = (tr.outcome == 'timeout').mean() * 100
    print(f"  Timeout orani       : %{to:.0f}")

    for s in ("SHORT", "LONG"):
        sub = tr[tr.side == s]
        if len(sub):
            sw = (sub.net_R > 0).mean() * 100
            print(f"    {s:<6}: {len(sub):>3} islem | win %{sw:.0f} | net {sub.net_R.mean():+.3f}R")

    if n < 100:
        se = 1.96 * np.sqrt(wr * (1 - wr) / n)
        print(f"\n  ! ORNEKLEM YETERSIZ (n={n} < 100).")
        print(f"    Kazanma orani %95 guven araligi: %{max(0,(wr-se))*100:.1f} - %{min(1,(wr+se))*100:.1f}")
        print(f"    Bu genislikte bir aralikta 'karli' hukmu verilemez.")

    print_funnel(rej)
    return dict(n=n, wr=wr, exp_net=exp_net, pf=pf, dd=dd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance_vision")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--csv")
    ap.add_argument("--min-rr", type=float, default=DEFAULTS["min_rr"])
    ap.add_argument("--lookback", type=int, default=DEFAULTS["lookback"])
    ap.add_argument("--risk", type=float, default=DEFAULTS["risk_frac"])
    ap.add_argument("--side", default="both", choices=["short", "long", "both"])
    ap.add_argument("--htf", type=int, default=1)
    ap.add_argument("--max-conf-cost", type=float, default=1.0)
    ap.add_argument("--oos", action="store_true", help="veriyi ikiye bol, IS/OOS karsilastir")
    ap.add_argument("--save")
    args = ap.parse_args()

    tf_hours = {"15m": .25, "30m": .5, "1h": 1, "2h": 2, "4h": 4, "12h": 12, "1d": 24}
    if args.tf not in tf_hours:
        sys.exit(f"Desteklenmeyen timeframe: {args.tf}")
    tfh = tf_hours[args.tf]

    if args.csv:
        df = load_csv(args.csv)
    else:
        sym = args.symbol.replace("/", "")
        print(f"Veri indiriliyor: {sym} {args.tf} ({args.days} gun)")
        df = fetch_binance(sym, args.tf, args.days)
    if df.empty:
        sys.exit("Veri alinamadi.")
    print(f"Veri: {len(df)} mum | {df.dt.iloc[0].date()} -> {df.dt.iloc[-1].date()}")

    p = dict(DEFAULTS)
    p.update(min_rr=args.min_rr, lookback=args.lookback, risk_frac=args.risk,
             htf_mult=args.htf, max_conf_cost=args.max_conf_cost, side=args.side)

    tr, rej = run_backtest(df, tfh, p)
    report(tr, p, f"TUM VERI  ({args.symbol} {args.tf}, R:R>={p['min_rr']}, side={p['side']})", rej)

    if args.oos:
        half = len(df) // 2
        d1, d2 = df.iloc[:half].reset_index(drop=True), df.iloc[half:].reset_index(drop=True)
        t1, r1 = run_backtest(d1, tfh, p)
        t2, r2 = run_backtest(d2, tfh, p)
        s1 = report(t1, p, "IN-SAMPLE (ilk yari)", r1)
        s2 = report(t2, p, "OUT-OF-SAMPLE (ikinci yari)", r2)
        print(f"\n{'=' * 62}\n  TUTARLILIK DEGERLENDIRMESI\n{'=' * 62}")
        if s1 and s2:
            print(f"  IS  beklenti: {s1['exp_net']:+.3f}R  (n={s1['n']})")
            print(f"  OOS beklenti: {s2['exp_net']:+.3f}R  (n={s2['n']})")
            if s1["exp_net"] > 0 and s2["exp_net"] > 0:
                print("  -> Iki yari da POZITIF. Umut verici, ancak orneklem kucukse\n"
                      "     hala kesin degildir.")
            elif s1["exp_net"] > 0 > s2["exp_net"]:
                print("  -> IS pozitif, OOS NEGATIF. Bu tipik OVERFIT isaretidir;\n"
                      "     kurulum saglam DEGILDIR.")
            else:
                print("  -> En az bir yari negatif. Kurulum dogrulanmadi.")
        else:
            print("  Yarilardan birinde islem uretilmedi; karsilastirma yapilamadi.")

    print("\nUYARI: Sonuclar kayma (slippage) icermez ve gecmis performans\n"
          "gelecegi garanti etmez. Bu bir yatirim tavsiyesi degildir.")

    if args.save and not tr.empty:
        tr.to_csv(args.save, index=False)
        print(f"Islemler kaydedildi: {args.save}")


if __name__ == "__main__":
    main()
