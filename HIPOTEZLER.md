# ON KAYITLI HIPOTEZLER

Kural: Bir bulgu once buraya yazilir, sonra YENI veriyle test edilir. Ayni veriden
hem bulunup hem uygulanan hicbir sey kural olamaz. Her hipotez icin gerekli
asgari yeni islem sayisi ve karar esigi onceden yazilir.

## Durum ozeti (2026-09-12, 155 dolmus islem, yeniden oynatma ile)

- Brut: +0.02R/islem (%95 GA [-0.32, +0.39]); net (ucret+%0.05 kayma): -0.11R
- Kar tek haftadan (ISO 33, +25R). En iyi 5 islem cikinca toplam -21.9R.
- 1h: +0.17R brut, +0.04R net. Uc parcada +0.60 / -0.20 / +0.12 -> kanitlanmamis.
- 4h: -0.40R (20 islem). Long/short: iki yarida isaret degistiriyor.
- Sonuc: mevcut veri, stratejinin maliyet sonrasi pozitif oldugunu DESTEKLEMIYOR.

## H1 - Trende karsi sinyaller (acik, 2026-09-12)

**Gozlem:** Coinin 1D trendinin TERSINE acilan sinyaller +0.64R (33 islem, GA
[-0.27, +1.55]); trend yonundekiler -0.45R (77 islem, GA [-0.84, -0.06]).
Iki yarida da ayni yonde. ~6 kirilim arasinda bulundu -> coklu test riski yuksek.

**Test:** Bu tarihten sonra dolan islemlerde `trend_uyumu()` kirilimi
(Telegram icgoru raporunda "Yon x 1D trend" satiri).

**Karar esigi:** En az 60 yeni "trende karsi" + 60 "trend yonunde" islem.
Fark yine ayni yonde ve trende-karsi net R'nin %95 GA'si sifiri dislarsa
-> trend yonundeki sinyalleri kapatma dusunulebilir. Aksi halde hipotez duser.

## H2 - 1h dilimi (acik)

**Karar esigi:** Duzeltilmis takiple 100 yeni 1h islemi. Net R GA'si sifiri
dislamiyorsa 1h de "kanitlanmamis" kalir; sistem testnet'te kalmaya devam eder.

## Dusen hipotezler

- "4h +1.40R" (devir dokumani): yeniden oynatmada -0.40R. Olcum hatasiydi.
- "SHORT daha iyi" / "LONG daha iyi": her ikisi de yarilar arasinda ters dondu.
- "BTC ayi rejimi +1.14R": tamamen 1. yaridan; zamanla karisik, anlamsiz.
