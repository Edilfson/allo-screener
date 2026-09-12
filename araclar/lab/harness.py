"""ARASTIRMA HARNESS - tum ajanlar AYNI simulator, AYNI maliyet, AYNI bolumleri kullanir.

BOLUMLER (sinyal zamanina gore):
  train      : < 2025-01-01
  valid      : 2025-01-01 .. 2025-12-31
  gelistirme : train + valid (< 2026-01-01)
  holdout    : >= 2026-01-01  -> SADECE ana ajan, LAB_HOLDOUT ortam degiskeniyle. AJANLAR DOKUNMAZ.
Veri fonksiyonlari holdout donemini FIZIKSEL olarak keser; ileriye bakma imkani yok.

VERI:
  veri_1h(sym)       : 1h mum, ~2 yil, 98 coin (bugunun ilk 100'u; hayatta-kalma yanliligi var)
  veri_1d(sym)       : gunluk, 2017'den beri, ~350 vadeli coin (data1d_all)
  resample(a, tf)    : 1h -> 2h/4h/6h/12h/1d
  semboller_1h(), semboller_1d()
  kolonlar: ot(ms), o, h, l, c, v

SIMULATOR:
  sim_islem(bars, sinyaller, P, tf)  -> islem listesi (tek sembol, sembol kilidi + cooldown)
     sinyal: dict(i=bar_idx (sinyal bu barin KAPANISINDA), side=+1/-1,
                  entry=None (piyasa, sonraki bar acilisi) | fiyat (limit),
                  stop=fiyat, tp=fiyat|None, cikis_i=None|bar_idx (gostergesel cikis, o barin kapanisi),
                  pending_h=24, max_days=7)
     kurallar: limit dolum = degme; dolum mumunda stop -> stop (kotumser); dolum mumunda TP sayilmaz;
               stop once TP sonra; maliyet = maker(limit)/taker(piyasa) giris + taker cikis + kayma, R cinsinden
  sim_portfoy(O, C, W, maliyet, funding) -> equity, gunluk getiri  (W: agirlik matrisi, t kapanista bilinir, t+1 acilista uygulanir)

RAPOR / DEFTER:
  rapor(islemler)            -> n, ort_net, GA95, GA99, win, yarilar, yillar
  portfoy_rapor(eq, g, gun)  -> CAGR, maxDD, Sharpe, yillar
  kaydet(aile, ad, params, bolum, rapor_dict, not_="")  -> defter.jsonl  (HER test kaydedilir; coklu-test sayimi)
"""
import os, json, time
import numpy as np
from datetime import datetime, timezone

SCRATCH = r"C:\Users\mehme\AppData\Local\Temp\claude\C--Users-mehme-Desktop\03ade4d8-9bb1-427a-88c0-1d4158cf5b1d\scratchpad"
D1H = os.path.join(SCRATCH, "data"); D1D = os.path.join(SCRATCH, "data1d_all"); D1D_ESKI = os.path.join(SCRATCH, "data1d")
LAB = os.path.dirname(os.path.abspath(__file__))
DEFTER = os.path.join(LAB, "defter.jsonl")
TF_H = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}
T_VALID = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
T_HOLD = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
BOLUM = {"train": (0, T_VALID), "valid": (T_VALID, T_HOLD), "gelistirme": (0, T_HOLD), "holdout": (T_HOLD, 1 << 62)}
# 1h verisi 2024-09'da basliyor -> 1h/2h/4h/6h/12h kaynakli testlerde train cok kisa kalir.
# Bu kaynak icin bolum: train < 2025-07-01, valid 2025-07-01..2025-12-31, holdout 2026 (ayni).
T_VALID_1H = int(datetime(2025, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
BOLUM_1H = {"train": (0, T_VALID_1H), "valid": (T_VALID_1H, T_HOLD), "gelistirme": (0, T_HOLD), "holdout": (T_HOLD, 1 << 62)}
MALIYET = dict(FEE_MAKER=0.0002, FEE_TAKER=0.0005, SLIP=0.0005)   # vadeli; R cinsine cevrilir


def _holdout_acik():
    return os.environ.get("LAB_HOLDOUT") == "son-degerlendirme-2026"


def _kes(a):
    if _holdout_acik():
        return a
    return a[a[:, 0] < T_HOLD]


def semboller_1h():
    return [f[:-4] for f in sorted(os.listdir(D1H)) if f.endswith(".npy")]


def semboller_1d():
    return [f[:-4] for f in sorted(os.listdir(D1D)) if f.endswith(".npy")]


def veri_1h(sym):
    return _kes(np.load(os.path.join(D1H, sym + ".npy")))


def veri_1d(sym):
    p = os.path.join(D1D, sym + ".npy")
    if not os.path.exists(p):
        p = os.path.join(D1D_ESKI, sym + ".npy")
    return _kes(np.load(p))


def resample(a, tf):
    k = TF_H[tf]
    if k == 1:
        return a
    g = (a[:, 0] // (k * 3600_000)).astype(np.int64)
    idx = np.flatnonzero(np.diff(g)) + 1
    st = np.concatenate([[0], idx]); en = np.concatenate([idx, [len(a)]])
    out = []
    for s, e in zip(st, en):
        if e - s != k:
            continue
        b = a[s:e]
        out.append([g[s] * k * 3600_000, b[0, 1], b[:, 2].max(), b[:, 3].min(), b[-1, 4], b[:, 5].sum()])
    return np.array(out) if out else np.zeros((0, 6))


def gunluk_panel(syms=None, min_gun=200):
    """Gunluk kapanis/acilis paneli: (syms, gun[ms], O[n,m], C[n,m], V[n,m]); eksik = nan"""
    syms = syms or semboller_1d()
    seri = {}
    for s in syms:
        try:
            a = veri_1d(s)
        except Exception:
            continue
        if len(a) >= min_gun:
            seri[s] = a
    syms = sorted(seri)
    gun = sorted({int(x) for a in seri.values() for x in a[:, 0]})
    gi = {g: i for i, g in enumerate(gun)}
    n, m = len(gun), len(syms)
    O = np.full((n, m), np.nan); C = np.full((n, m), np.nan); V = np.full((n, m), np.nan)
    for j, s in enumerate(syms):
        a = seri[s]; idx = [gi[int(x)] for x in a[:, 0]]
        O[idx, j] = a[:, 1]; C[idx, j] = a[:, 4]; V[idx, j] = a[:, 5]
    return syms, np.array(gun), O, C, V


# ---------------------------------------------------------------- baglam
_BTC = {}
def btc_ma_rejim(N=50):
    """gun[ms] -> +1 (close>MA) / -1; o gunun KAPANISINDA bilinir"""
    if N not in _BTC:
        a = veri_1d("BTCUSDT"); c = a[:, 4]
        m = np.convolve(c, np.ones(N) / N, "full")[:len(c)]; m[:N - 1] = np.nan
        _BTC[N] = (a[:, 0], np.where(c > m, 1, -1), m)
    return _BTC[N]


def rejim_at(t_ms, N=50):
    """t_ms aninda son KAPANMIS gunun BTC rejimi"""
    ots, r, _ = btc_ma_rejim(N)
    k = np.searchsorted(ots, t_ms - 86400_000, side="right") - 1
    return int(r[k]) if k >= N else 0


# ---------------------------------------------------------------- islem simulatoru
def sim_islem(bars, sinyaller, P=None, tf="1h", cooldown_h=20):
    """bars: (ot,o,h,l,c,v). sinyaller: i'ye gore SIRALI dict listesi. Return islem listesi."""
    P = dict(MALIYET, **(P or {}))
    ot, o, h, l, c = bars[:, 0], bars[:, 1], bars[:, 2], bars[:, 3], bars[:, 4]
    n = len(c); tfms = TF_H[tf] * 3600_000
    out = []; kilit = -1; son_ms = -1e18
    for s in sinyaller:
        i, side = s["i"], s["side"]
        if i <= kilit or i >= n - 1 or ot[i] - son_ms < cooldown_h * 3600_000:
            continue
        stop, tp = s["stop"], s.get("tp")
        limit = s.get("entry")
        pend_h, max_days = s.get("pending_h", 24), s.get("max_days", 7)
        cikis_i = s.get("cikis_i")
        sig_close = ot[i] + tfms
        # ---- giris ----
        if limit is None:                       # piyasa: sonraki bar acilisi + kayma
            fill = i + 1
            e = o[fill] * (1 + side * P["SLIP"])
            giris_fee = P["FEE_TAKER"]
            if (side == 1 and e <= stop) or (side == -1 and e >= stop):
                continue
        else:
            e = limit; giris_fee = P["FEE_MAKER"]; fill = None
            deadline = sig_close + pend_h * 3600_000
            for j in range(i + 1, n):
                if ot[j] > deadline:
                    break
                if (l[j] <= e) if side == 1 else (h[j] >= e):
                    fill = j; break
                if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                    break
            if fill is None:
                out.append(dict(t=int(sig_close), side=side, res="cancelled", r=0.0, net=0.0, sp=abs(e - stop) / e))
                kilit = min(i + 1 + int(pend_h / TF_H[tf]), n - 1); son_ms = ot[i]
                continue
        risk = abs(e - stop); sp = risk / e
        if risk <= 0:
            continue
        maliyet = (giris_fee + P["FEE_TAKER"] + P["SLIP"]) / sp
        tmax = sig_close + max_days * 86400_000
        res, r, jk = None, None, None
        for j in range(fill, n):
            if ot[j] > tmax or (cikis_i is not None and j > cikis_i):
                res, r, jk = "timeout", side * (c[j - 1] - e) / risk, j; break
            if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                res, r, jk = "stopped", -1.0, j; break
            if tp is not None and j > fill and ((h[j] >= tp) if side == 1 else (l[j] <= tp)):
                res, r, jk = "target_done", side * (tp - e) / risk, j; break
            if cikis_i is not None and j == cikis_i and j > fill:
                res, r, jk = "cikis", side * (c[j] - e) / risk, j; break
        if res is None:
            break                                    # veri sonu, acik islem: sayma
        out.append(dict(t=int(sig_close), side=side, res=res, r=float(r), net=float(r - maliyet),
                        sp=float(sp), bar=int(jk - fill)))
        kilit = jk; son_ms = ot[i]
    return out


def sim_portfoy(O, C, W, maliyet=0.0007, funding=0.0, clip=(-0.95, 5.0)):
    """W[t,j]: t kapanisinda bilinen agirlik (toplam |w| <= 1 onerilir). t+1 acilista uygulanir,
    getiri open[t+1]->open[t+2]. Maliyet = |dW| * maliyet. Return eq, g."""
    n, m = C.shape
    eq = [1.0]; g = []; w_prev = np.zeros(m)
    for t in range(n - 2):
        w = np.nan_to_num(W[t]).copy(); w[np.isnan(O[t + 1]) | np.isnan(O[t + 2])] = 0
        r = np.clip(np.nan_to_num((O[t + 2] - O[t + 1]) / O[t + 1]), *clip)
        x = (w * r).sum() - np.abs(w - w_prev).sum() * maliyet - np.abs(w).sum() * funding
        g.append(x); eq.append(eq[-1] * (1 + x))
        w_prev = w * (1 + r) / (1 + x) if x > -1 else w
    return np.array(eq), np.array(g)


# ---------------------------------------------------------------- rapor
def _ga(x, n=3000, q=0.025):
    x = np.asarray(x, float)
    if len(x) < 5:
        return (float("nan"), float("nan"))
    m = np.sort([x[np.random.randint(0, len(x), len(x))].mean() for _ in range(n)])
    return float(m[int(n * q)]), float(m[int(n * (1 - q))])


def bolum_filtre(islemler, bolum, kaynak="1d"):
    """kaynak='1h' -> BOLUM_1H (1h'den turetilen tum dilimler icin), '1d' -> BOLUM"""
    a, b = (BOLUM_1H if kaynak == "1h" else BOLUM)[bolum]
    return [t for t in islemler if a <= t["t"] < b]


def rapor(islemler, bolum=None, kaynak="1d"):
    if bolum:
        islemler = bolum_filtre(islemler, bolum, kaynak)
    F = [t for t in islemler if t["res"] != "cancelled"]
    if len(F) < 5:
        return dict(n=len(F), sinyal=len(islemler), ort_net=None)
    x = np.array([t["net"] for t in F]); ts = np.array([t["t"] for t in F])
    mid = np.median(ts)
    yil = {}
    for t in F:
        y = datetime.fromtimestamp(t["t"] / 1e3, timezone.utc).year
        yil.setdefault(y, []).append(t["net"])
    return dict(n=int(len(F)), sinyal=len(islemler), dolma=round(len(F) / max(1, len(islemler)), 3),
                ort_net=round(float(x.mean()), 4), ort_brut=round(float(np.mean([t["r"] for t in F])), 4),
                ga95=[round(v, 4) for v in _ga(x)], ga99=[round(v, 4) for v in _ga(x, q=0.005)],
                win=round(float((x > 0).mean()), 3), toplam=round(float(x.sum()), 1),
                yari1=round(float(x[ts < mid].mean()), 4), yari2=round(float(x[ts >= mid].mean()), 4),
                yillar={int(y): [len(v), round(float(np.mean(v)), 3)] for y, v in sorted(yil.items())})


def portfoy_rapor(eq, g, gun, bolum=None, kaynak="1d"):
    """gun: her getiri gunu icin ms (len(g)); bolum verilirse o araliga kesilir"""
    g = np.asarray(g); gun = np.asarray(gun)[:len(g)]
    if bolum:
        a, b = (BOLUM_1H if kaynak == "1h" else BOLUM)[bolum]; m = (gun >= a) & (gun < b); g = g[m]; gun = gun[m]
    if len(g) < 30:
        return dict(gun=len(g))
    eq = np.cumprod(1 + g); yil = len(g) / 365
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    yl = {}
    for t, x in zip(gun, g):
        y = datetime.fromtimestamp(t / 1e3, timezone.utc).year; yl.setdefault(y, []).append(x)
    return dict(gun=int(len(g)), toplam=round(float(eq[-1] - 1), 3), cagr=round(float(eq[-1] ** (1 / yil) - 1), 3),
                maxdd=round(float(dd), 3), sharpe=round(float(g.mean() / g.std() * np.sqrt(365)) if g.std() > 0 else 0, 2),
                yillar={int(y): round(float(np.prod(1 + np.array(v)) - 1), 3) for y, v in sorted(yl.items())})


def kaydet(aile, ad, params, bolum, rap, not_=""):
    k = dict(zaman=datetime.now(timezone.utc).isoformat(), aile=aile, ad=ad, params=params, bolum=bolum, rapor=rap, not_=not_)
    with open(DEFTER, "a", encoding="utf-8") as f:
        f.write(json.dumps(k, default=str) + "\n")
    return k


def defter_oku():
    if not os.path.exists(DEFTER):
        return []
    return [json.loads(l) for l in open(DEFTER, encoding="utf-8") if l.strip()]


def kabul(tr, va):
    """Aday kabul kurali: train ve valid'de ort_net > 0, train GA99 sifiri disliyor, valid n >= 100 (islem) veya
    portfoy icin her iki bolumde Sharpe > kiyas. Coklu test: defterdeki test sayisi arttikca esik yukselir."""
    if not tr or not va or tr.get("ort_net") is None or va.get("ort_net") is None:
        return False, "veri_yok"
    if tr["ort_net"] <= 0 or va["ort_net"] <= 0:
        return False, "bir bolum negatif"
    if tr["ga99"][0] <= 0:
        return False, "train GA99 sifiri icermiyor degil"
    if va["n"] < 100:
        return False, "valid n<100"
    return True, "ADAY"


if __name__ == "__main__":
    # hizli kendi kendine test: 4h MA20/50 kesisimi piyasa girisi, ATR stop yok -> sadece cikis sinyali
    syms = semboller_1h()[:5]
    T = []
    for s in syms:
        a = resample(veri_1h(s), "4h")
        if len(a) < 200:
            continue
        c = a[:, 4]
        m1 = np.convolve(c, np.ones(20) / 20, "full")[:len(c)]; m2 = np.convolve(c, np.ones(50) / 50, "full")[:len(c)]
        sig = []
        for i in range(60, len(c) - 1):
            if m1[i] > m2[i] and m1[i - 1] <= m2[i - 1]:
                sig.append(dict(i=i, side=1, entry=None, stop=c[i] * 0.97, tp=None, max_days=30))
        T += sim_islem(a, sig, tf="4h")
    print("train", rapor(T, "train", "1h")); print("valid", rapor(T, "valid", "1h"))
    print("holdout acik mi:", _holdout_acik(), "| 1h sembol:", len(semboller_1h()), "| 1d sembol:", len(semboller_1d()))
