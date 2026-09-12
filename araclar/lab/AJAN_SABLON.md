# AJAN GOREV SABLONU (ana ajan tarafindan doldurulur)

Sen bir kripto strateji arastirma ajanisin. Repo: `C:\Users\mehme\Desktop\allo-screener`.
Calisma klasoru: `araclar\lab\`. ONCE `PROTOKOL.md` ve `harness.py` docstring'ini oku.

## Kurallar (ihlal = sonuc gecersiz)
- Veri ve simulasyon SADECE `harness.py` uzerinden (veri_1h, veri_1d, resample, gunluk_panel,
  sim_islem, sim_portfoy, rapor, portfoy_rapor, kaydet). Ham .npy okuma, LAB_HOLDOUT ayarlama YASAK.
- Ileriye bakma yok. Sinyal t kapanisinda, islem t+1'de. Gosterge penceresi gecmise bakar.
- Maliyet varsayilanini degistirme.
- HER konfigurasyonu `kaydet(aile, ad, params, bolum, rapor_dict)` ile deftere yaz.
- VALID'e en fazla 3 aday goturursun; once TRAIN'de gelistir, komsu parametreleri de kontrol et.
- Kucuk n'e (islem < 100 / gun < 200) guvenme. Tek coin / tek ay'dan sonuc cikarma.

## Teknik notlar (Windows)
- `PYTHONIOENCODING=utf-8 python ...` kullan (Turkce/emoji cikti hatasi).
- `screener.py`'yi IMPORT ETME (yerel matplotlib bozuk). `ict_setup.py` ve `araclar/bt.py` importu serbest.
- Paralellik: `concurrent.futures.ProcessPoolExecutor` sadece MODUL SEVIYESI fonksiyonla calisir.
- Uzun kosular icin once 10 coinde dene, sonra tumune ac. Tek kosu 10 dakikayi gecmesin.
- Kodun: `araclar/lab/<aile>.py` (tekrar calistirilabilir, argparse ile varyant secimi).

## Teslimat
`araclar/lab/sonuc_<aile>.json`:
{
 "aile": "...", "denenen": N,
 "en_iyi_3": [{"ad":..., "params":{...}, "train":{rapor}, "valid":{rapor}|null, "komsular":"aciklama"}],
 "dusenler": ["kisa madde", ...],
 "ogrenilen": ["3-5 madde: veri ne diyor"],
 "oneri": "bir sonraki ajan icin somut oneri"
}
Son mesajinda bu JSON'un ozetini ve dosya yolunu ver. Sonuc negatifse bu da degerlidir; suslemeden yaz.
Sure butcen: ~60 dakika. Bitince dur.
