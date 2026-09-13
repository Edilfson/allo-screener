"""mtf_yapi - coklu zaman dilimi YAPISAL uyum + yapidan belirlenen stop/TP (ONCEDEN KAYITLI tek hipotez ailesi).

Veri: harness.veri_1h + resample (98 coin), kaynak="1h". TRAIN < 2025-07-01, VALID 2025-07..12 (en fazla 3 aday, tek sefer).
Parametre taramasi YOK: 12 varyant = yon {LONG,SHORT} x cikis {A,C,D} x filtre {F1,F2} + kontroller.

NEDENSELLIK: 1h bar i kapanisinda (T = ot1h[i] + 1h) sadece kapanis zamani <= T olan 4h ve 1d barlar.
Swing teyidi ict2/bt kurali: idx < i - 3 (k=3 bar gecikmeli).

KURULUM (LONG; SHORT tam simetrik)
 1 GUNLUK YON : son kapanmis gunde coin close > MA50 ve MA50[k] > MA50[k-10]. F2: + BTC gunluk close > MA50.
 2 4h BOLGE   : ict2._yapi (BOS + displacement>=1.5xATR14 + FVG + OB) ilk tespit edildigi 4h barda olusur;
                bolge = [min(OB lo, FVG lo), max(OB hi, FVG hi)] (iki araligin kapsayan birlesimi).
                Gecerli: olusum kapanisindan sonra <= 30 adet 4h bar (30*4h) VE hic bir kapanmis 4h close bolge altinda degil.
                Ilk dokunus (1h low <= bolge hi) sonrasi 12 adet 1h bar (dokunus bari dahil) tetik penceresi.
 3 1h TETIK   : (a) supurme: 1h low < son teyitli 1h swing low VE ayni bar veya sonraki bar o seviye ustunde kapanis
                (b) BOS: 1h close > son teyitli 1h swing high. Hangisi once. Giris: sonraki 1h acilis (sim_islem entry=None).
                Bolge basina en fazla 1 sinyal (ilk tetik bolgeyi tuketir; atlanan sinyal de tuketir).
 4 STOP       : min(bolge lo, pencere [dokunus..tetik] en dusuk 1h low) - max(0.25*ATR14_1h, %0.10*close). %0.4..%8 disi -> atla.
 5 TP ADAYLARI: son 3 teyitli 4h swing high; onceki (son kapanmis) gunun high'i; son 2 teyitli 1d swing high;
                girisin ustundeki en yakin gecerli 4h DUSUS bolgesinin alt kenari. RR = (seviye - ref)/risk, ref = tetik bari close.
   A: RR>=2 olan ilk seviye (RR<=10 degilse / yoksa atla), TP = seviye*(1-0.001).
   C: %50 A seviyesinde, stop BE (giris) ; kalan %50 bir sonraki (daha uzak) aday seviyede, yoksa 2*RR_A.
   D: KONTROL sabit 3R (A'nin sinyal kumesi, ayni giris ve stop).
   Zaman asimi 5 gun. Maliyet harness varsayilani, sembol kilidi + 20 saat cooldown.

KONTROLLER (>= 20 seed, coin basina ayni sayida sinyal)
 PL1: ayni gunluk yon + ayni 4h bolge pencereleri, tetik yerine pencereden RASTGELE 1h bar; ayni stop/TP insasi.
 PL2: ayni gunluk yon, bolge yok, RASTGELE 1h bar; stop% = stratejinin (coin, yon, filtre) stop% listesinden rastgele;
      TP adaylari o bardaki gercek yapisal seviyeler.
 ABLASYON F0: gunluk yon filtresi olmadan ayni strateji (gunluk yonun katkisi; aday olamaz).
 TP BILGISI: A/D/C ilk bacak TP isabet orani - ort(p0), p0 = 1/(1+RR_gercek) (maliyetsiz rastgele yuruyus).

Kullanim:
  PYTHONIOENCODING=utf-8 python mtf_yapi.py --asama uret --coin 10 --onek duman
  PYTHONIOENCODING=utf-8 python mtf_yapi.py --asama hepsi --coin 98 --onek tam
  asamalar: uret | test (sim esdegerlik + nedensellik) | analiz | valid --adaylar "LONG|F1|A,..." | hepsi
"""
import os, sys, json, argparse, hashlib, time, pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
ARACLAR = os.path.dirname(LAB)
sys.path.insert(0, LAB); sys.path.insert(0, ARACLAR)
import harness as H
import bt
import ict2

AILE = "mtf_yapi"
P = dict(bt.P0)                 # SWING_K=3, DISP_MULT=1.5, LOOKBACK_BOS=40, OB_ARAMA=12, MAX_BOS_YAS=20 (DEGISMEZ)
HOUR = 3600_000; DAY = 86400_000; H4 = 4 * HOUR
ZONE_MAX_4H = 30; WIN_1H = 12
STOP_MIN, STOP_MAX = 0.004, 0.08
BUF_ATR, BUF_PCT = 0.25, 0.001
RR_MIN, RR_MAX, TP_PAY = 2.0, 10.0, 0.001
MAX_DAYS = 5; COOLDOWN_H = 20
N_SEED = 20
WINS = (-1.5, 3.0)
ARA = os.path.join(H.SCRATCH, "mtf_yapi")
SIDE_AD = {1: "LONG", -1: "SHORT"}
CIKISLAR = ("A", "C", "D")
T_TRAIN0 = int(datetime(2024, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
PERIYOT = {"train": (0, H.T_VALID_1H), "valid": (H.T_VALID_1H, H.T_HOLD)}
# satir kolonlari
COL = dict(t=0, side=1, res=2, r=3, net=4, sp=5, rr=6, hit=7, rrg=8, p0=9)
RES = {"stopped": 0, "target_done": 1, "timeout": 2, "tp1_be": 3, "tp1_tp2": 4, "tp1_timeout": 5}
ALPHA_BONF = 0.05 / 12


# ================================================================ baglam
def _sma(x, N=50):
    m = np.convolve(x, np.ones(N) / N, "full")[:len(x)]
    m[:N - 1] = np.nan
    return m


def gunluk_coin(sym, a1h):
    try:
        a = H.veri_1d(sym)
        if len(a) >= 70:
            return a
    except Exception:
        pass
    return H.resample(a1h, "1d")


def bolgeler(g4, b4):
    """4h ICT bolgeleri (ict2._yapi), (yon, bos_i) ile tekillestirilmis; ilk tespit bari = olusum"""
    c4 = g4["c"]; n4 = len(c4); Z = []; seen = set()
    for j in range(80, n4):
        for side in (1, -1):
            y = ict2._yapi(g4, j, side, P)
            if y is None:
                continue
            key = (side, int(y["bos_i"]))
            if key in seen:
                continue
            seen.add(key)
            lo = float(min(y["ob"][0], y["fvg"][0])); hi = float(max(y["ob"][1], y["fvg"][1]))
            form = float(b4[j, 0] + H4)
            bad = np.flatnonzero(c4[j + 1:] < lo) if side == 1 else np.flatnonzero(c4[j + 1:] > hi)
            inval = float(b4[j + 1 + bad[0], 0] + H4) if len(bad) else np.inf
            Z.append(dict(side=side, j=j, bos_i=int(y["bos_i"]), lo=lo, hi=hi, ob=[float(v) for v in y["ob"]],
                          fvg=[float(v) for v in y["fvg"]], form=form, exp=form + ZONE_MAX_4H * H4, inval=inval))
    return Z


def baglam(a1h, d1c, btc):
    n = len(a1h)
    b4 = H.resample(a1h, "4h")
    g1 = bt.hazirla(a1h, P); g4 = bt.hazirla(b4, P); gd = bt.hazirla(d1c, P)
    T = a1h[:, 0] + HOUR
    j4 = np.searchsorted(b4[:, 0] + H4, T, side="right") - 1
    kd = np.searchsorted(d1c[:, 0] + DAY, T, side="right") - 1
    cd = d1c[:, 4]; ma = _sma(cd); ma10 = np.full(len(ma), np.nan); ma10[10:] = ma[:-10]
    with np.errstate(invalid="ignore"):
        ddir = np.where((cd > ma) & (ma > ma10), 1, np.where((cd < ma) & (ma < ma10), -1, 0))
    coin_dir = np.where(kd >= 0, ddir[np.maximum(kd, 0)], 0)
    bc = btc[:, 4]; bma = _sma(bc)
    with np.errstate(invalid="ignore"):
        bdir = np.where(bc > bma, 1, np.where(bc < bma, -1, 0))
    kb = np.searchsorted(btc[:, 0] + DAY, T, side="right") - 1
    btc_dir = np.where(kb >= 0, bdir[np.maximum(kb, 0)], 0)
    ar = np.arange(n)
    SL = np.full(n, np.nan); SH = np.full(n, np.nan)
    if len(g1["sl_idx"]):
        m = np.searchsorted(g1["sl_idx"], ar - P["SWING_K"]); ok = m >= 1
        SL[ok] = g1["l"][g1["sl_idx"][m[ok] - 1]]
    if len(g1["sh_idx"]):
        m = np.searchsorted(g1["sh_idx"], ar - P["SWING_K"]); ok = m >= 1
        SH[ok] = g1["h"][g1["sh_idx"][m[ok] - 1]]
    Z = bolgeler(g4, b4) if len(b4) > 90 else []
    Zarr = {}
    for s in (1, -1):
        zz = [z for z in Z if z["side"] == s]
        Zarr[s] = {k: np.array([z[k] for z in zz], float) for k in ("form", "exp", "inval", "lo", "hi")}
    return dict(n=n, a=a1h, ot=a1h[:, 0], T=T, o=a1h[:, 1], h=a1h[:, 2], l=a1h[:, 3], c=a1h[:, 4], atr=g1["atr"],
                SL=SL, SH=SH, b4=b4, g4=g4, gd=gd, d1c=d1c, j4=j4, kd=kd, coin_dir=coin_dir, btc_dir=btc_dir,
                Z=Z, Zarr=Zarr)


def filtre_ok(ctx, side, filt):
    if filt == 0:
        return np.ones(ctx["n"], bool)
    ok = ctx["coin_dir"] == side
    if filt == 2:
        ok &= ctx["btc_dir"] == side
    return ok


# ================================================================ stop / TP
def tp_adaylar(ctx, i, side, ref):
    T = ctx["T"][i]; lv = []
    j = int(ctx["j4"][i]); g4 = ctx["g4"]
    if j >= 0:
        idx = g4["sh_idx"] if side == 1 else g4["sl_idx"]; px = g4["h"] if side == 1 else g4["l"]
        m = np.searchsorted(idx, j - P["SWING_K"])
        for s in idx[max(0, m - 3):m]:
            lv.append((float(px[s]), "4h_swing"))
    k = int(ctx["kd"][i]); gd = ctx["gd"]
    if k >= 0:
        lv.append((float(gd["h"][k] if side == 1 else gd["l"][k]), "onceki_gun"))
        idx = gd["sh_idx"] if side == 1 else gd["sl_idx"]; px = gd["h"] if side == 1 else gd["l"]
        m = np.searchsorted(idx, k - P["SWING_K"])
        for s in idx[max(0, m - 2):m]:
            lv.append((float(px[s]), "1d_swing"))
    Za = ctx["Zarr"][-side]
    if len(Za["form"]):
        ok = (Za["form"] <= T) & (T < Za["inval"]) & (T <= Za["exp"])
        if side == 1:
            e = Za["lo"][ok]; e = e[e > ref]
            if len(e):
                lv.append((float(e.min()), "4h_dusus_bolge_alt"))
        else:
            e = Za["hi"][ok]; e = e[e < ref]
            if len(e):
                lv.append((float(e.max()), "4h_yukselis_bolge_ust"))
    lv = [x for x in lv if side * (x[0] - ref) > 0]
    lv.sort(key=lambda x: side * (x[0] - ref))
    return lv


def stop_hesap(ctx, i, side, i_t, zone):
    a = ctx["atr"][i]; c = ctx["c"][i]
    if np.isnan(a):
        return None, "atr_yok", None
    buf = max(BUF_ATR * a, BUF_PCT * c)
    if side == 1:
        wl = float(ctx["l"][i_t:i + 1].min())
        base = min(zone["lo"], wl); src = "4h_bolge_alt" if zone["lo"] <= wl else "1h_pencere_dip"
        stop = base - buf
    else:
        wh = float(ctx["h"][i_t:i + 1].max())
        base = max(zone["hi"], wh); src = "4h_bolge_ust" if zone["hi"] >= wh else "1h_pencere_tepe"
        stop = base + buf
    sp = side * (c - stop) / c
    if sp < STOP_MIN:
        return None, "stop_dar", None
    if sp > STOP_MAX:
        return None, "stop_genis", None
    return float(stop), None, src


def plan_kur(ctx, i, side, stop):
    """giriste bilinen her seyle A/C/D cikis planlari"""
    ref = float(ctx["c"][i]); risk = side * (ref - stop)
    if risk <= 0:
        return None, "risk_yok"
    cands = tp_adaylar(ctx, i, side, ref)
    sel = None
    for q, (lvl, src) in enumerate(cands):
        if side * (lvl - ref) / risk >= RR_MIN:
            sel = q; break
    if sel is None:
        return None, "tp_yok"
    lvl1, src1 = cands[sel]; rr1 = side * (lvl1 - ref) / risk
    if rr1 > RR_MAX:
        return None, "tp_rr_ust"
    tp1 = lvl1 * (1 - side * TP_PAY)
    tp2 = None
    for lvl2, src2 in cands[sel + 1:]:
        if side * (lvl2 - lvl1) > 0:
            tp2 = lvl2 * (1 - side * TP_PAY); rr2 = side * (lvl2 - ref) / risk; break
    if tp2 is None:
        rr2 = 2 * rr1; tp2 = ref + side * rr2 * risk; src2 = "2xRR_A"
    return dict(i=int(i), side=side, stop=float(stop), sp=float(risk / ref), ref=ref,
                tp=float(tp1), rr=float(rr1), src=src1, tp2=float(tp2), rr2=float(rr2), src2=src2,
                tp3=float(ref + side * 3 * risk), cands=[(round(a, 8), s, round(side * (a - ref) / risk, 3)) for a, s in cands]), None


# ================================================================ strateji
def _tetik(ctx, i, side, i_t):
    l, h, c, SL, SH = ctx["l"], ctx["h"], ctx["c"], ctx["SL"], ctx["SH"]
    if side == 1:
        L = SL[i]
        if l[i] < L and c[i] > L:
            return "supurme"
        if i - 1 >= i_t:
            L0 = SL[i - 1]
            if l[i - 1] < L0 and c[i] > L0:
                return "supurme_sonraki_bar"
        if c[i] > SH[i]:
            return "bos"
    else:
        L = SH[i]
        if h[i] > L and c[i] < L:
            return "supurme"
        if i - 1 >= i_t:
            L0 = SH[i - 1]
            if h[i - 1] > L0 and c[i] < L0:
                return "supurme_sonraki_bar"
        if c[i] < SL[i]:
            return "bos"
    return None


def strateji(ctx, side, filt, a_ms, b_ms, son_bar_ok=False):
    """Return plans (i sirali, tekil i), pencereler [(zone_idx, i_t, eligible np.array)], sayac"""
    T, l, h = ctx["T"], ctx["l"], ctx["h"]; n = ctx["n"]
    fok = filtre_ok(ctx, side, filt)
    cnt = dict(bolge=0, dokunus=0, pencere=0, tetik=0, tetik_yok=0, atr_yok=0, stop_dar=0, stop_genis=0,
               risk_yok=0, tp_yok=0, tp_rr_ust=0, ayni_bar=0, gecerli=0)
    tet_tur = {}
    plans, pen = [], []; used = set()
    for zi, z in enumerate(ctx["Z"]):
        if z["side"] != side or z["form"] >= b_ms or z["exp"] < a_ms:
            continue
        cnt["bolge"] += 1
        s = int(np.searchsorted(ctx["ot"], z["form"], side="left"))
        end = min(int(np.searchsorted(T, z["exp"], side="right")), int(np.searchsorted(T, z["inval"], side="left")), n)
        if s >= end:
            continue
        seg = (l[s:end] <= z["hi"]) if side == 1 else (h[s:end] >= z["lo"])
        if not seg.any():
            continue
        i_t = s + int(np.argmax(seg)); cnt["dokunus"] += 1
        wbars = np.arange(i_t, min(i_t + WIN_1H, end))
        el = wbars[fok[wbars] & (T[wbars] >= a_ms) & (T[wbars] < b_ms)]
        if not son_bar_ok:
            el = el[el < n - 1]
        if len(el) == 0:
            continue
        cnt["pencere"] += 1
        pen.append((zi, i_t, el))
        tetik = None
        for i in el:
            tt = _tetik(ctx, int(i), side, i_t)
            if tt:
                tetik = (int(i), tt); break
        if tetik is None:
            cnt["tetik_yok"] += 1; continue
        i, tt = tetik; cnt["tetik"] += 1
        stop, seb, ssrc = stop_hesap(ctx, i, side, i_t, z)
        if stop is None:
            cnt[seb] += 1; continue
        pl, seb = plan_kur(ctx, i, side, stop)
        if pl is None:
            cnt[seb] += 1; continue
        if i in used:
            cnt["ayni_bar"] += 1; continue
        used.add(i)
        pl.update(zone=zi, z_lo=z["lo"], z_hi=z["hi"], z_form=z["form"], z_ob=z["ob"], z_fvg=z["fvg"],
                  i_t=int(i_t), tetik=tt, stop_src=ssrc)
        tet_tur[tt] = tet_tur.get(tt, 0) + 1
        plans.append(pl); cnt["gecerli"] += 1
    plans.sort(key=lambda p: p["i"])
    cnt["tetik_tur"] = tet_tur
    return plans, pen, cnt


def pl1_sinyaller(ctx, side, pen, n_hedef, rng):
    out, used = [], set()
    if not pen or n_hedef <= 0:
        return out
    cap = n_hedef * 60 + 500; den = 0
    order = rng.permutation(len(pen)); pos = 0
    while len(out) < n_hedef and den < cap:
        if pos >= len(order):
            order = rng.permutation(len(pen)); pos = 0
        zi, i_t, el = pen[order[pos]]; pos += 1; den += 1
        i = int(el[rng.integers(len(el))])
        if i in used:
            continue
        stop, seb, ssrc = stop_hesap(ctx, i, side, i_t, ctx["Z"][zi])
        if stop is None:
            continue
        pl, seb = plan_kur(ctx, i, side, stop)
        if pl is None:
            continue
        used.add(i); out.append(pl)
    out.sort(key=lambda p: p["i"])
    return out


def pl2_sinyaller(ctx, side, filt, a_ms, b_ms, sps, n_hedef, rng):
    out, used = [], set()
    if n_hedef <= 0 or not len(sps):
        return out
    T = ctx["T"]; n = ctx["n"]
    ok = filtre_ok(ctx, side, filt) & (T >= a_ms) & (T < b_ms) & ~np.isnan(ctx["atr"])
    ok[n - 1:] = False; ok[:100] = False
    el = np.flatnonzero(ok)
    if not len(el):
        return out
    cap = n_hedef * 60 + 500; den = 0
    while len(out) < n_hedef and den < cap:
        den += 1
        i = int(el[rng.integers(len(el))])
        if i in used:
            continue
        sp = float(sps[rng.integers(len(sps))])
        stop = ctx["c"][i] * (1 - side * sp)
        pl, seb = plan_kur(ctx, i, side, stop)
        if pl is None:
            continue
        used.add(i); out.append(pl)
    out.sort(key=lambda p: p["i"])
    return out


# ================================================================ simulasyon
def sim_kismi(bars, sinyaller, P_=None, cooldown_h=COOLDOWN_H):
    """harness.sim_islem kurallarinin (piyasa girisi) birebir kopyasi + opsiyonel %50 kismi cikis.
    tp2 None -> tek bacak (sim_islem ile ayni sonuc olmali). tp2 varsa: bacak1 %50 tp'de, stop girise (BE);
    bacak2 %50 tp2 / BE / zaman asimi. Bacak2 ve BE, TP1 barindan SONRAKI bardan itibaren kontrol edilir
    (ayni bar icinde sira bilinmez). Stop once TP sonra; dolum mumunda TP sayilmaz, stop gecerli."""
    PP = dict(H.MALIYET, **(P_ or {}))
    ot, o, h, l, c = bars[:, 0], bars[:, 1], bars[:, 2], bars[:, 3], bars[:, 4]
    n = len(c); tfms = HOUR
    out = []; kilit = -1; son_ms = -1e18
    for s in sinyaller:
        i, side = s["i"], s["side"]
        if i <= kilit or i >= n - 1 or ot[i] - son_ms < cooldown_h * 3600_000:
            continue
        stop, tp, tp2 = s["stop"], s.get("tp"), s.get("tp2")
        max_days = s.get("max_days", 7)
        sig_close = ot[i] + tfms
        fill = i + 1
        e = o[fill] * (1 + side * PP["SLIP"])
        if (side == 1 and e <= stop) or (side == -1 and e >= stop):
            continue
        risk = abs(e - stop); sp = risk / e
        if risk <= 0:
            continue
        maliyet = (PP["FEE_TAKER"] + PP["FEE_TAKER"] + PP["SLIP"]) / sp
        tmax = sig_close + max_days * 86400_000
        res, r, jk = None, None, None
        if tp2 is None:
            for j in range(fill, n):
                if ot[j] > tmax:
                    res, r, jk = "timeout", side * (c[j - 1] - e) / risk, j; break
                if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                    res, r, jk = "stopped", -1.0, j; break
                if tp is not None and j > fill and ((h[j] >= tp) if side == 1 else (l[j] <= tp)):
                    res, r, jk = "target_done", side * (tp - e) / risk, j; break
            if res is None:
                break
            out.append(dict(t=int(sig_close), side=side, res=res, r=float(r), net=float(r - maliyet),
                            sp=float(sp), bar=int(jk - fill)))
        else:
            r1 = None; j1 = None; r2 = None
            for j in range(fill, n):
                if ot[j] > tmax:
                    rt = side * (c[j - 1] - e) / risk
                    if r1 is None:
                        r1 = r2 = rt; res = "timeout"
                    else:
                        r2 = rt; res = "tp1_timeout"
                    jk = j; break
                if r1 is None:
                    if (l[j] <= stop) if side == 1 else (h[j] >= stop):
                        r1 = r2 = -1.0; res = "stopped"; jk = j; break
                    if j > fill and ((h[j] >= tp) if side == 1 else (l[j] <= tp)):
                        r1 = side * (tp - e) / risk; j1 = j
                    continue
                if j > j1:
                    if (l[j] <= e) if side == 1 else (h[j] >= e):
                        r2 = 0.0; res = "tp1_be"; jk = j; break
                    if (h[j] >= tp2) if side == 1 else (l[j] <= tp2):
                        r2 = side * (tp2 - e) / risk; res = "tp1_tp2"; jk = j; break
            if res is None:
                break
            r = 0.5 * r1 + 0.5 * r2
            net = 0.5 * (r1 - maliyet) + 0.5 * (r2 - maliyet)     # maliyet her cikis bacagi icin ayri
            out.append(dict(t=int(sig_close), side=side, res=res, r=float(r), net=float(net), sp=float(sp),
                            bar=int(jk - fill), r1=float(r1), r2=float(r2)))
        kilit = jk; son_ms = ot[i]
    return out


def _satirlar(trades, plans, cikis, ctx):
    pm = {int(ctx["T"][p["i"]]): p for p in plans}
    rows = []; det = []
    for tr in trades:
        if tr["res"] == "cancelled":
            continue
        p = pm[tr["t"]]; side = p["side"]
        tp = p["tp3"] if cikis == "D" else p["tp"]
        e = p["stop"] / (1 - side * tr["sp"])
        rrg = side * (tp - e) / abs(e - p["stop"])
        p0 = 1.0 / (1.0 + rrg) if rrg > 0 else 1.0
        hit = 1.0 if tr["res"] in ("target_done", "tp1_be", "tp1_tp2", "tp1_timeout") else 0.0
        rr = 3.0 if cikis == "D" else p["rr"]
        rows.append([tr["t"], side, RES[tr["res"]], tr["r"], tr["net"], tr["sp"], rr, hit, rrg, p0])
        det.append((tr, p, e))
    return np.array(rows, float).reshape(-1, len(COL)), det


def _sim_hepsi(ctx, plans, cikislar=CIKISLAR):
    out = {}
    if not plans:
        return out, None
    base = [dict(i=p["i"], side=p["side"], entry=None, stop=p["stop"], max_days=MAX_DAYS) for p in plans]
    for cx in cikislar:
        if cx == "A":
            sg = [dict(b, tp=p["tp"]) for b, p in zip(base, plans)]
            tr = H.sim_islem(ctx["a"], sg, tf="1h", cooldown_h=COOLDOWN_H)
        elif cx == "D":
            sg = [dict(b, tp=p["tp3"]) for b, p in zip(base, plans)]
            tr = H.sim_islem(ctx["a"], sg, tf="1h", cooldown_h=COOLDOWN_H)
        else:
            sg = [dict(b, tp=p["tp"], tp2=p["tp2"]) for b, p in zip(base, plans)]
            tr = sim_kismi(ctx["a"], sg)
        out[cx] = _satirlar(tr, plans, cx, ctx)
    return out, base


def _esdeger_test(ctx, plans):
    """D planlari: sim_kismi(tp2=None) == harness.sim_islem ; Return (karsilastirilan islem, uyusmayan)"""
    if not plans:
        return 0, 0
    sg = [dict(i=p["i"], side=p["side"], entry=None, stop=p["stop"], tp=p["tp3"], max_days=MAX_DAYS) for p in plans]
    a = H.sim_islem(ctx["a"], sg, tf="1h", cooldown_h=COOLDOWN_H)
    b = sim_kismi(ctx["a"], sg)
    uy = 0 if len(a) == len(b) else abs(len(a) - len(b)) + 1
    for x, y in zip(a, b):
        if x["t"] != y["t"] or x["res"] != y["res"] or abs(x["r"] - y["r"]) > 1e-9 or abs(x["net"] - y["net"]) > 1e-9:
            uy += 1
    return len(a), uy


def _seed(*parts):
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


_BTC = {}


def _btc():
    if "b" not in _BTC:
        _BTC["b"] = H.veri_1d("BTCUSDT")
    return _BTC["b"]


def _isci(arg):
    sym, onek, bolum, n_seed, secim = arg
    yol = os.path.join(ARA, onek, f"{sym}.pkl")
    if os.path.exists(yol):
        return sym, "var"
    try:
        a_ms, b_ms = PERIYOT[bolum]
        a1h = H.veri_1h(sym)
        if len(a1h) < 1000:
            pickle.dump(None, open(yol, "wb")); return sym, "kisa"
        ctx = baglam(a1h, gunluk_coin(sym, a1h), _btc())
        R, atla, ornek = {}, {}, []
        esit = [0, 0]
        for side in (1, -1):
            for filt in (0, 1, 2):
                vs = [cx for cx in CIKISLAR if (SIDE_AD[side], filt, cx) in secim]
                if not vs:
                    continue
                plans, pen, cnt = strateji(ctx, side, filt, a_ms, b_ms)
                atla[f"{SIDE_AD[side]}|F{filt}"] = cnt
                sims, _ = _sim_hepsi(ctx, plans, vs)
                for cx, (rows, det) in sims.items():
                    R[f"S|{SIDE_AD[side]}|F{filt}|{cx}|0"] = rows
                    if cx == "A" and filt == 1:
                        for tr, p, e in det:
                            ornek.append(dict(sym=sym, t=tr["t"], zaman=datetime.fromtimestamp(tr["t"] / 1e3, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                                              yon=SIDE_AD[side], res=tr["res"], net=round(tr["net"], 3), giris=round(e, 8),
                                              stop=round(p["stop"], 8), stop_kaynak=p["stop_src"], tp=round(p["tp"], 8),
                                              tp_kaynak=p["src"], rr_plan=round(p["rr"], 2), tetik=p["tetik"],
                                              bolge=[round(p["z_lo"], 8), round(p["z_hi"], 8)], bolge_ob=p["z_ob"], bolge_fvg=p["z_fvg"],
                                              bolge_olusum=datetime.fromtimestamp(p["z_form"] / 1e3, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                                              tetik_ref_close=p["ref"], aday_seviyeler=p["cands"]))
                if filt == 0 or not plans:
                    continue
                if "D" in vs:
                    k, u = _esdeger_test(ctx, plans); esit[0] += k; esit[1] += u
                sps = np.array([p["sp"] for p in plans])
                for sd in range(n_seed):
                    rng = np.random.default_rng(_seed(sym, side, filt, "PL1", sd))
                    p1 = pl1_sinyaller(ctx, side, pen, len(plans), rng)
                    s1, _ = _sim_hepsi(ctx, p1, vs)
                    for cx, (rows, _) in s1.items():
                        R[f"PL1|{SIDE_AD[side]}|F{filt}|{cx}|{sd}"] = rows
                    if sd == 0 and "D" in vs:
                        k, u = _esdeger_test(ctx, p1); esit[0] += k; esit[1] += u
                    rng = np.random.default_rng(_seed(sym, side, filt, "PL2", sd))
                    p2 = pl2_sinyaller(ctx, side, filt, a_ms, b_ms, sps, len(plans), rng)
                    s2, _ = _sim_hepsi(ctx, p2, vs)
                    for cx, (rows, _) in s2.items():
                        R[f"PL2|{SIDE_AD[side]}|F{filt}|{cx}|{sd}"] = rows
        pickle.dump(dict(sym=sym, R=R, atla=atla, ornek=ornek, esit=esit), open(yol, "wb"))
        return sym, "ok"
    except Exception as ex:
        import traceback
        return sym, f"HATA {ex} {traceback.format_exc()[-600:]}"


def tum_secim(f0=True):
    s = set()
    for sd in ("LONG", "SHORT"):
        for f in ((0, 1, 2) if f0 else (1, 2)):
            for cx in CIKISLAR:
                s.add((sd, f, cx))
    return s


def uret(syms, onek, bolum="train", isci=8, n_seed=N_SEED, secim=None):
    os.makedirs(os.path.join(ARA, onek), exist_ok=True)
    secim = secim or tum_secim()
    t0 = time.time(); durum = {}
    args = [(s, onek, bolum, n_seed, secim) for s in syms]
    with ProcessPoolExecutor(max_workers=isci) as ex:
        for k, (sym, st) in enumerate(ex.map(_isci, args, chunksize=1)):
            durum[sym] = st
            if st.startswith("HATA"):
                print(sym, st, flush=True)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(syms)} coin, {time.time()-t0:.0f}s", flush=True)
    print(f"uret bitti {onek} {bolum}: {time.time()-t0:.0f}s, hata={sum(1 for v in durum.values() if v.startswith('HATA'))}", flush=True)
    return durum


# ================================================================ nedensellik
def nedensellik(syms, n_test=50, seed=5):
    """veriyi sinyal barinda kes -> ayni bolge / stop / TP (A,C,D) / aday seviyeler cikmali"""
    rng = np.random.default_rng(seed)
    btc = _btc(); havuz = []
    for sym in syms:
        a1h = H.veri_1h(sym)
        if len(a1h) < 1000:
            continue
        d1c = gunluk_coin(sym, a1h)
        ctx = baglam(a1h, d1c, btc)
        for side in (1, -1):
            for filt in (0, 1, 2):
                plans, _, _ = strateji(ctx, side, filt, *PERIYOT["train"])
                havuz += [(sym, side, filt, p) for p in plans]
    secim = [havuz[k] for k in rng.choice(len(havuz), min(n_test, len(havuz)), replace=False)]
    ayni, farkli, detay = 0, 0, []
    cache = {}
    for sym, side, filt, p in secim:
        if sym not in cache:
            a1h = H.veri_1h(sym); cache = {sym: (a1h, gunluk_coin(sym, a1h))}
        a1h, d1c = cache[sym]
        i = p["i"]; T = a1h[i, 0] + HOUR
        a_t = a1h[:i + 1]; d_t = d1c[d1c[:, 0] + DAY <= T]; b_t = btc[btc[:, 0] + DAY <= T]
        ctx_t = baglam(a_t, d_t, b_t)
        pl_t, _, _ = strateji(ctx_t, side, filt, 0, T + 1, son_bar_ok=True)
        q = [x for x in pl_t if x["i"] == i]
        ok = False
        if q:
            x = q[0]
            ok = (abs(x["z_lo"] - p["z_lo"]) < 1e-12 and abs(x["z_hi"] - p["z_hi"]) < 1e-12 and x["z_form"] == p["z_form"]
                  and abs(x["stop"] - p["stop"]) < 1e-9 * p["ref"] and abs(x["tp"] - p["tp"]) < 1e-9 * p["ref"]
                  and abs(x["tp2"] - p["tp2"]) < 1e-9 * p["ref"] and abs(x["tp3"] - p["tp3"]) < 1e-9 * p["ref"]
                  and x["cands"] == p["cands"] and x["tetik"] == p["tetik"])
        if ok:
            ayni += 1
        else:
            farkli += 1
            detay.append(dict(sym=sym, i=i, side=side, tam=dict(stop=p["stop"], tp=p["tp"], c=p["cands"]),
                              kesik=(dict(stop=q[0]["stop"], tp=q[0]["tp"], c=q[0]["cands"]) if q else None)))
    return dict(test=len(secim), ayni=ayni, farkli=farkli, detay=detay[:5])


# ================================================================ istatistik
def _w(x):
    return np.clip(x, *WINS)


def _top1(net):
    if len(net) < 20 or net.sum() == 0:
        return None
    xs = np.sort(net)[::-1]; k = max(1, int(len(net) * 0.01))
    return float(xs[:k].sum() / abs(net.sum()))


def _ay(t):
    return np.asarray(t, "int64").astype("datetime64[ms]").astype("datetime64[M]").astype(int)


def ay_blok_fark(tS, xS, tP, xP, B=4000, seed=1):
    """eslesmis ay-blok bootstrap: ayni aylar iki taraftan birlikte cekilir. mean(S)-mean(P)"""
    if len(xS) < 10 or len(xP) < 10:
        return None
    mS, mP = _ay(tS), _ay(tP)
    aylar = np.unique(np.concatenate([mS, mP])); ai = {a: k for k, a in enumerate(aylar)}; K = len(aylar)
    sS = np.zeros(K); nS = np.zeros(K); sP = np.zeros(K); nP = np.zeros(K)
    np.add.at(sS, [ai[a] for a in mS], xS); np.add.at(nS, [ai[a] for a in mS], 1)
    np.add.at(sP, [ai[a] for a in mP], xP); np.add.at(nP, [ai[a] for a in mP], 1)
    cnt = np.random.default_rng(seed).multinomial(K, np.ones(K) / K, size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (cnt @ sS) / (cnt @ nS) - (cnt @ sP) / (cnt @ nP)
    d = d[np.isfinite(d)]
    q = lambda a: round(float(np.percentile(d, a)), 4)
    return dict(fark=round(float(np.mean(xS) - np.mean(xP)), 4), ay=int(K), ga95=[q(2.5), q(97.5)],
                ga996=[q(100 * ALPHA_BONF / 2), q(100 - 100 * ALPHA_BONF / 2)])


def _boot_ort(x, B=4000, seed=2):
    x = np.asarray(x, float)
    if len(x) < 10:
        return None
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def tp_bilgi(rows):
    if len(rows) < 10:
        return None
    hit = rows[:, COL["hit"]]; p0 = rows[:, COL["p0"]]; res = rows[:, COL["res"]]
    coz = res != RES["timeout"]
    return dict(n=int(len(rows)), isabet=round(float(hit.mean()), 4), p0_ort=round(float(p0.mean()), 4),
                fark=round(float((hit - p0).mean()), 4), ga95=_boot_ort(hit - p0),
                zaman_asimi_orani=round(float((res == RES["timeout"]).mean()), 3),
                cozulen_fark=(round(float((hit[coz] - p0[coz]).mean()), 4) if coz.sum() >= 10 else None))


def ozet_rows(rows, t_sinir=None):
    if len(rows) < 5:
        return dict(n=int(len(rows)))
    net = rows[:, COL["net"]]; t = rows[:, COL["t"]]
    mid = np.median(t)
    o = dict(n=int(len(rows)), ort_brut=round(float(rows[:, COL["r"]].mean()), 4), ort_net=round(float(net.mean()), 4),
             wins3_net=round(float(_w(net).mean()), 4), win=round(float((net > 0).mean()), 3),
             ort_rr=round(float(rows[:, COL["rr"]].mean()), 3), top1_pay=(round(_top1(net), 3) if _top1(net) is not None else None),
             yari1=round(float(net[t < mid].mean()), 4), yari2=round(float(net[t >= mid].mean()), 4),
             sonuc_dagilimi={k: int((rows[:, COL["res"]] == v).sum()) for k, v in RES.items() if (rows[:, COL["res"]] == v).any()})
    if t_sinir is not None:
        kat = np.searchsorted(t_sinir, t, side="right")
        kw = [round(float(_w(net[kat == k]).mean()), 4) if (kat == k).any() else None for k in range(4)]
        o["kat_wins3"] = kw; o["kat_n"] = [int((kat == k).sum()) for k in range(4)]
        o["kat_pozitif"] = int(sum(1 for v in kw if v is not None and v > 0))
    return o


def yukle(onek):
    D = {}; atla = {}; ornek = []; esit = [0, 0]
    d = os.path.join(ARA, onek)
    for f in sorted(os.listdir(d)):
        if not f.endswith(".pkl"):
            continue
        x = pickle.load(open(os.path.join(d, f), "rb"))
        if x is None:
            continue
        for k, v in x["R"].items():
            if len(v):
                D.setdefault(k, []).append(v)
        for k, c in x["atla"].items():
            A = atla.setdefault(k, {})
            for kk, vv in c.items():
                if isinstance(vv, dict):
                    B = A.setdefault(kk, {})
                    for a, b in vv.items():
                        B[a] = B.get(a, 0) + b
                else:
                    A[kk] = A.get(kk, 0) + vv
        ornek += x["ornek"]; esit[0] += x["esit"][0]; esit[1] += x["esit"][1]
    D = {k: np.concatenate(v) for k, v in D.items()}
    return D, atla, ornek, esit


def _havuz(D, tur, sd, f, cx, n_seed):
    per = [D.get(f"{tur}|{sd}|F{f}|{cx}|{s}", np.zeros((0, len(COL)))) for s in range(n_seed)]
    return np.concatenate(per) if per else np.zeros((0, len(COL))), per


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def analiz(onek, bolum="train", n_seed=N_SEED, kaydet=True, varyantlar=None):
    D, atla, ornek, esit = yukle(onek)
    V = {}
    for sd in ("LONG", "SHORT"):
        for f in (1, 2):
            for cx in CIKISLAR:
                ad = f"{sd}|F{f}|{cx}"
                if varyantlar and ad not in varyantlar:
                    continue
                S = D.get(f"S|{ad}|0", np.zeros((0, len(COL))))
                ts = S[:, COL["t"]]
                sinir = np.quantile(ts, [0.25, 0.5, 0.75]) if len(ts) >= 8 else None
                r = ozet_rows(S, sinir)
                cnt = atla.get(f"{sd}|F{f}", {})
                r["sinyal_akisi"] = cnt
                r["simde_kilit_cooldown_atlanan"] = int(cnt.get("gecerli", 0) - len(S))
                r["tp_bilgi"] = tp_bilgi(S)
                # harness.rapor (GA95/GA99)
                tr = [dict(t=int(x[COL["t"]]), res="x", net=float(x[COL["net"]]), r=float(x[COL["r"]])) for x in S]
                hr = H.rapor(tr, bolum, "1h")
                r["harness_rapor"] = {k: hr.get(k) for k in ("n", "ort_net", "ga95", "ga99", "win", "yillar")}
                for pl in ("PL1", "PL2"):
                    Pp, per = _havuz(D, pl, sd, f, cx, n_seed)
                    seed_w = [float(_w(x[:, COL["net"]]).mean()) for x in per if len(x) >= 5]
                    r[pl] = dict(ozet=ozet_rows(Pp), seed_wins3_p5_50_95=([round(float(np.percentile(seed_w, q)), 4) for q in (5, 50, 95)] if seed_w else None),
                                 tp_bilgi=tp_bilgi(Pp),
                                 fark_wins3=ay_blok_fark(ts, _w(S[:, COL["net"]]), Pp[:, COL["t"]], _w(Pp[:, COL["net"]])),
                                 fark_net=ay_blok_fark(ts, S[:, COL["net"]], Pp[:, COL["t"]], Pp[:, COL["net"]]),
                                 fark_tp_isabet_eksi_p0=ay_blok_fark(ts, S[:, COL["hit"]] - S[:, COL["p0"]], Pp[:, COL["t"]],
                                                                     Pp[:, COL["hit"]] - Pp[:, COL["p0"]]))
                # aday kurali
                f1 = r["PL1"]["fark_wins3"]; f2 = r["PL2"]["fark_wins3"]
                sart = dict(wins3_pozitif=bool(r.get("wins3_net") is not None and r["wins3_net"] > 0),
                            pl1_ga996_alt_pozitif=bool(f1 and f1["ga996"][0] > 0),
                            pl2_ga996_alt_pozitif=bool(f2 and f2["ga996"][0] > 0),
                            kat_3_4=bool(r.get("kat_pozitif", 0) >= 3),
                            top1_lt_50=bool(r.get("top1_pay") is not None and r["top1_pay"] < 0.5),
                            n_150=bool(r["n"] >= 150))
                r["aday_sartlari"] = sart; r["ADAY"] = all(sart.values())
                V[ad] = r
                if kaydet:
                    H.kaydet(AILE, f"{ad}|strateji", dict(yon=sd, filtre=f"F{f}", cikis=cx, sabitler=_sabitler()), bolum,
                             {k: r[k] for k in r if k not in ("PL1", "PL2")}, "onceden kayitli hipotez; 12 varyanttan biri")
                    for pl in ("PL1", "PL2"):
                        H.kaydet(AILE, f"{ad}|{pl}", dict(yon=sd, filtre=f"F{f}", cikis=cx, seed=n_seed,
                                                          kontrol="rastgele tetik bari (bolge penceresi)" if pl == "PL1" else "rastgele bar, bolge yok"),
                                 bolum, r[pl], "plasebo kontrol")
    # ablasyon F0 + bilesen katkilari
    AB = {}
    for sd in ("LONG", "SHORT"):
        for cx in CIKISLAR:
            ad0 = f"{sd}|F0|{cx}"
            S0 = D.get(f"S|{ad0}|0")
            if S0 is None:
                continue
            o0 = ozet_rows(S0); o0["sinyal_akisi"] = atla.get(f"{sd}|F0", {})
            S1 = D.get(f"S|{sd}|F1|{cx}|0", np.zeros((0, len(COL))))
            o0["F1_eksi_F0_wins3"] = ay_blok_fark(S1[:, COL["t"]], _w(S1[:, COL["net"]]), S0[:, COL["t"]], _w(S0[:, COL["net"]]))
            AB[ad0] = o0
            if kaydet:
                H.kaydet(AILE, f"{ad0}|ablasyon_gunluk_yon_yok", dict(yon=sd, filtre="F0", cikis=cx), bolum, o0,
                         "ablasyon: gunluk yon filtresi kaldirildi (aday olamaz)")
    return V, AB, ornek, esit, atla, D


def _sabitler():
    return dict(ZONE_MAX_4H=ZONE_MAX_4H, WIN_1H=WIN_1H, STOP_MIN=STOP_MIN, STOP_MAX=STOP_MAX, BUF_ATR=BUF_ATR, BUF_PCT=BUF_PCT,
                RR_MIN=RR_MIN, RR_MAX=RR_MAX, TP_PAY=TP_PAY, MAX_DAYS=MAX_DAYS, COOLDOWN_H=COOLDOWN_H, P="bt.P0 (ict2._yapi)")


def bilesen_katkisi(V, AB, D, n_seed=N_SEED):
    tab = {}
    for sd in ("LONG", "SHORT"):
        for f in (1, 2):
            A = D.get(f"S|{sd}|F{f}|A|0", np.zeros((0, len(COL)))); Dd = D.get(f"S|{sd}|F{f}|D|0", np.zeros((0, len(COL))))
            for cx in CIKISLAR:
                ad = f"{sd}|F{f}|{cx}"
                if ad not in V:
                    continue
                r = V[ad]
                e = dict(strateji_wins3=r.get("wins3_net"), strateji_net=r.get("ort_net"), n=r["n"],
                         tetik_1h_katkisi_S_eksi_PL1=r["PL1"]["fark_wins3"],
                         bolge_4h_katkisi_S_eksi_PL2=r["PL2"]["fark_wins3"])
                if f == 1 and f"{sd}|F0|{cx}" in AB:
                    e["gunluk_yon_katkisi_F1_eksi_F0"] = AB[f"{sd}|F0|{cx}"]["F1_eksi_F0_wins3"]
                    e["F0_wins3"] = AB[f"{sd}|F0|{cx}"].get("wins3_net")
                if f == 2:
                    S1 = D.get(f"S|{sd}|F1|{cx}|0", np.zeros((0, len(COL)))); S2 = D.get(f"S|{sd}|F2|{cx}|0", np.zeros((0, len(COL))))
                    e["btc_yon_katkisi_F2_eksi_F1"] = ay_blok_fark(S2[:, COL["t"]], _w(S2[:, COL["net"]]), S1[:, COL["t"]], _w(S1[:, COL["net"]]))
                if cx in ("A", "C"):
                    X = A if cx == "A" else D.get(f"S|{sd}|F{f}|C|0", np.zeros((0, len(COL))))
                    e[f"yapisal_tp_katkisi_{cx}_eksi_D3R"] = ay_blok_fark(X[:, COL["t"]], _w(X[:, COL["net"]]), Dd[:, COL["t"]], _w(Dd[:, COL["net"]]))
                e["tp_bilgi_S_PL1_PL2"] = [r["tp_bilgi"] and r["tp_bilgi"]["fark"], r["PL1"]["tp_bilgi"] and r["PL1"]["tp_bilgi"]["fark"],
                                           r["PL2"]["tp_bilgi"] and r["PL2"]["tp_bilgi"]["fark"]]
                tab[ad] = e
    return tab


def ornek_sec(ornek, seed=3):
    rng = np.random.default_rng(seed)
    kaz = [o for o in ornek if o["res"] == "target_done"]
    kay = [o for o in ornek if o["res"] == "stopped"]
    sec = lambda L: [L[k] for k in sorted(rng.choice(len(L), min(3, len(L)), replace=False))] if L else []
    return dict(kazanan=sec(kaz), kaybeden=sec(kay),
                not_="TRAIN, cikis A, filtre F1 (LONG+SHORT havuzu), rastgele secim seed=3; giris = gercek dolum (acilis+kayma)")


# ================================================================ ana
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", default="hepsi")
    ap.add_argument("--coin", type=int, default=98)
    ap.add_argument("--onek", default="tam")
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--seed", type=int, default=N_SEED)
    ap.add_argument("--adaylar", default="")
    ap.add_argument("--kaydet", type=int, default=1)     # duman testinde 0 (deftere yazma)
    a = ap.parse_args()
    syms = H.semboller_1h()[:a.coin]
    cikti = os.path.join(ARA, f"{a.onek}_analiz.pkl")
    if a.asama in ("uret", "hepsi"):
        uret(syms, a.onek, "train", a.isci, a.seed)
    if a.asama in ("test", "hepsi"):
        t0 = time.time()
        ned = nedensellik(syms[:25], 50)
        print("NEDENSELLIK:", json.dumps(ned)[:600], f"{time.time()-t0:.0f}s", flush=True)
        json.dump(ned, open(os.path.join(ARA, f"{a.onek}_nedensellik.json"), "w"))
    if a.asama in ("analiz", "hepsi"):
        V, AB, ornek, esit, atla, D = analiz(a.onek, "train", a.seed, a.kaydet == 1)
        tab = bilesen_katkisi(V, AB, D, a.seed)
        pickle.dump(dict(V=V, AB=AB, ornek=ornek, esit=esit, atla=atla, tab=tab), open(cikti, "wb"))
        print("SIM ESDEGERLIK (D: sim_kismi vs sim_islem): karsilastirilan", esit[0], "uyusmayan", esit[1])
        for ad, r in V.items():
            f1 = r["PL1"]["fark_wins3"]; f2 = r["PL2"]["fark_wins3"]; tb = r["tp_bilgi"] or {}
            print(f"{ad:14s} n={r['n']:4d} brut={r.get('ort_brut')} net={r.get('ort_net')} w3={r.get('wins3_net')} win={r.get('win')} "
                  f"rr={r.get('ort_rr')} top1={r.get('top1_pay')} kat={r.get('kat_wins3')} | PL1 w3={r['PL1']['ozet'].get('wins3_net')} "
                  f"d={f1 and f1['fark']} {f1 and f1['ga996']} | PL2 w3={r['PL2']['ozet'].get('wins3_net')} d={f2 and f2['fark']} {f2 and f2['ga996']} "
                  f"| tp-p0={tb.get('fark')} {tb.get('ga95')} ADAY={r['ADAY']}", flush=True)
        for ad, o in AB.items():
            print("ABL", ad, o.get("n"), o.get("wins3_net"), o.get("F1_eksi_F0_wins3"), flush=True)
    if a.asama == "valid" and a.adaylar:
        ads = [x for x in a.adaylar.split(",") if x][:3]
        sec = set()
        for x in ads:
            sd, f, cx = x.split("|"); sec.add((sd, int(f[1:]), cx))
        on = a.onek + "_valid"
        uret(syms, on, "valid", a.isci, a.seed, sec)
        V, AB, ornek, esit, atla, D = analiz(on, "valid", a.seed, True, set(ads))
        pickle.dump(dict(V=V), open(os.path.join(ARA, f"{on}_analiz.pkl"), "wb"))
        for ad, r in V.items():
            print("VALID", ad, json.dumps(_json({k: r[k] for k in ("n", "ort_net", "wins3_net", "kat_pozitif", "top1_pay", "ADAY")})),
                  r["PL1"]["fark_wins3"], r["PL2"]["fark_wins3"], flush=True)


if __name__ == "__main__":
    main()
