"""Liderlik tablosu: defter.jsonl + sonuc_*.json -> adaylar, kabul kontrolu, coklu-test sayaci."""
import os, json, glob, sys
import harness as H
LAB = H.LAB
D = H.defter_oku()
print(f"DEFTER: {len(D)} test | aileler: { {k: sum(1 for d in D if d['aile']==k) for k in sorted({d['aile'] for d in D})} }")
print("(coklu test: N test icinde en iyinin sansla GA99 gecme riski N ile buyur -> VALID/HOLDOUT tek sefer)\n")

def islem_satir(ad, r):
    if not r or r.get("ort_net") is None:
        return f"  {ad:<40} (veri yok)"
    return (f"  {ad:<40} n={r['n']:>5} net={r['ort_net']:+.3f} GA99[{r['ga99'][0]:+.3f},{r['ga99'][1]:+.3f}] "
            f"win=%{r['win']*100:.0f} yari={r['yari1']:+.2f}/{r['yari2']:+.2f}")

def port_satir(ad, r):
    if not r or "sharpe" not in r:
        return f"  {ad:<40} (veri yok)"
    return f"  {ad:<40} CAGR={r['cagr']*100:+.1f}% maxDD={r['maxdd']*100:.0f}% Sharpe={r['sharpe']:.2f} yillar={r.get('yillar')}"

for f in sorted(glob.glob(os.path.join(LAB, "sonuc_*.json"))):
    try:
        S = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print(f"== {os.path.basename(f)}: OKUNAMADI {e}"); continue
    print(f"== {S.get('aile', os.path.basename(f))} | denenen {S.get('denenen')} ==")
    for a in S.get("en_iyi_3", []):
        tr, va = a.get("train"), a.get("valid")
        satir = port_satir if (tr and "sharpe" in tr) else islem_satir
        print(f" * {a.get('ad')}  params={a.get('params')}")
        print(satir("   train", tr)); print(satir("   valid", va))
        if tr and va and "sharpe" not in (tr or {}):
            ok, neden = H.kabul(tr, va); print(f"   -> {'ADAY' if ok else 'RED'}: {neden}")
        print(f"   komsular: {a.get('komsular')}")
    for m in S.get("ogrenilen", []): print(f"   - {m}")
    print(f"   oneri: {S.get('oneri')}\n")
