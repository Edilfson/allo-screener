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

## H3 - BTC/ETH/BNB gunluk MA50 trend takibi (acik, 2026-09-12)

**Kanit (araclar/btc_uzun.py, 2017-2026):** MA10..MA200 tum varyantlar al-tutu
Sharpe'ta geciyor; MA50 IS ve OOS'ta CAGR ve maxDD'de al-tutu geciyor (RAPOR_2026-09-12.md).
Bu, ICT'nin aksine parametreye ve doneme dayanikli.

**Canli test:** trend_bot.py kagit portfoyu (trend_state.json `kagit`), esit agirlik,
maliyet %0.1. Yilda ~6-12 islem oldugu icin canli dogrulama YAVAS; asil kanit backtest.

**Karar esigi:** Kagit portfoy 12 ay sonra al-tuttan daha dusuk maxDD gostermeli.
Getiri farki gurultulu olacak; dususte koruma gorulmuyorsa hipotez zayiflar.

## H1 - Trende karsi sinyaller -> REDDEDILDI (2026-09-12, buyuk orneklem)

2 yil / 98 coin / 2881 islem: trende karsi -0.26R (GA sifiri disliyor), trend yonunde
-0.11R. Canlidaki 33 islemlik +0.64R gurultuydu. Asagidaki eski metin kayit icin duruyor.

### (eski) H1 - Trende karsi sinyaller (acik, 2026-09-12)

**Gozlem:** Coinin 1D trendinin TERSINE acilan sinyaller +0.64R (33 islem, GA
[-0.27, +1.55]); trend yonundekiler -0.45R (77 islem, GA [-0.84, -0.06]).
Iki yarida da ayni yonde. ~6 kirilim arasinda bulundu -> coklu test riski yuksek.

**Test:** Bu tarihten sonra dolan islemlerde `trend_uyumu()` kirilimi
(Telegram icgoru raporunda "Yon x 1D trend" satiri).

**Karar esigi:** En az 60 yeni "trende karsi" + 60 "trend yonunde" islem.
Fark yine ayni yonde ve trende-karsi net R'nin %95 GA'si sifiri dislarsa
-> trend yonundeki sinyalleri kapatma dusunulebilir. Aksi halde hipotez duser.

## H2 - 1h dilimi -> REDDEDILDI (2026-09-12)

Backtest 2065 islem: net -0.14R, GA [-0.23, -0.04]; iki yarida da negatif. 19 parametre
varyantinin hicbiri IS+OOS pozitif degil. ICT parametreyle kurtarilamaz.

## Dusen hipotezler

- "4h +1.40R" (devir dokumani): yeniden oynatmada -0.40R. Olcum hatasiydi.
- "SHORT daha iyi" / "LONG daha iyi": her ikisi de yarilar arasinda ters dondu.
- "BTC ayi rejimi +1.14R": tamamen 1. yaridan; zamanla karisik, anlamsiz.

## H4 - HOLDOUT ON KAYDI (2026-09-13, lab programi)
Holdout (2026-01-01..bugun) adaylari, degerlendirmeden ONCE ilan: A1_b3_ma50, A2_b3_ma50_vt30 (ana),
Y_yuruyen3_ma50_vt30, Y_yuruyen3_ma50. Kiyaslar: b3 al-tut, BTC al-tut, yuruyen top-3 al-tut.
Basari olcutu: holdout'ta Sharpe VE maxDD, kendi kiyasindan iyi. Beklenti (tsmom2): Sharpe 0.7-1.0 bile basari.
Sonradan aday EKLENMEZ. Betik: araclar/lab/holdout_degerlendir.py (LAB_HOLDOUT ile).
Elenen aileler (lab, 2026-09-13): kesitsel momentum, intraday kirilim, ict2 (rastgele kontrol), rejim/MR/sikisma.

### H4 SONUCU (holdout 2026-01-01..2026-09-12, 252 gun, 2026-09-13 tek sefer)
| aday | getiri | maxDD | Sharpe | kendi al-tutu (getiri/maxDD/Sharpe) |
|---|---|---|---|---|
| A1 b3 MA50 | +11.2% | 18.6% | 0.69 | -16.0% / 45.0% / -0.27 |
| A2 b3 MA50+vt30 | +13.5% | 14.7% | 0.94 | -16.0% / 45.0% / -0.27 |
| Y yuruyen3 MA50+vt30 | +17.9% | 14.7% | 1.10 | +7.7% / 45.0% / 0.47 |
| Y yuruyen3 MA50 | +33.0% | 18.6% | 1.27 | +7.7% / 45.0% / 0.47 |
Brut-eslenmis al-tut: A2 Sharpe 0.94 vs -0.27 (DD 14.7 vs 20.0); Y 1.10 vs 0.47 (14.7 vs 18.6).
KARAR: 4/4 aday olcutu gecti (Sharpe VE maxDD kiyastan iyi). Beklenti araligi (0.7-1.35) icinde.
Canli: trend_portfoy.py (vt20 cap1.0) + trend_testnet.py. Bundan sonra holdout verisi KAPALI; yeni
karar ancak canli kagit/testnet verisiyle (H3 esigi: 12 ay).

## H5 - Yuruyen top-5 + tek-coin tavani %20 (on kayit 2026-09-13, kesif_tavan)
**Kesif (gelistirme bolumu, holdout KULLANILMADI):** tavan tek basina olcek kucultme (gross-esit maxDD daha kotu);
gap riskini asil dusuren sepet genisligi (top-3 -> top-5: -90% sok kumulatif kayip 0.73 -> 0.51); top-5'te %20
tavan neredeyse bedava (-0.02 Sharpe). Ters kanit: 2024-25'te genislik zarar ettirdi (top3 +0.30 vs top5 +0.15).
**Canli test:** trend_portfoy.py 3. kagit portfoy `yuruyen5_tavan` (MA50 + vt20 + tavan 0.20).
**Karar esigi (12 ay):** H5 maxDD < Y maxDD VE Sharpe farki > -0.20 ise kalici; gercek -%40+ gecelik gap
yasanirsa o olaydaki darbe farki tek basina belirleyici. Beklenen: Sharpe ~ayni, maxDD 0.27 -> 0.24.

## 2026-09-13 ek tur: kullanici istegiyle 3 yeni aile (hepsi RED)
Protokol ayni: TRAIN <2025 (4h icin <2025-07) / VALID 2025 tek sefer / holdout kapali; trend-ici veya rastgele plasebo zorunlu.
- **ema_geri** (yukselis sonrasi EMA20-50 bandina geri cekilme + donus tetigi, 4h ve 1d, 76 konfig): 0 aday. Trend icinde
  rastgele mumdan anlamli iyi degil (en iyi fark GA95 alt siniri -0.12R); 4h'de ikinci yari 27/27 negatif. Kazanc trendde
  olmaktan geliyor, kurulum bir sey eklemiyor.
- **tersine** (kesitsel kisa vadeli tersine donus, gunluk, 144 konfig): 0 aday. L/S 64/64 negatif Sharpe; maliyetsiz bile ~0
  (uclar donmuyor); long-only pozitifligi BTC betasi, rastgele secimden kotu. Momentum (onceki tur) + tersine = altcoin
  kesit sinyalleri iki yonde de maliyet sonrasi olu.
- **funding** (spot long + perp short carry; funding yuzdeligi sinyali): carry TRAIN'de 22/23 gecti ama VALID 2025'te
  3 adayin 3'u olcutu gecemedi (hep acik -%1.2, esikli +%0.2, top10 -%1.0); getiri 2021 %20 -> 2023-25 nakit faizinin
  altinda (rejim bitmis). Funding sinyali 12 testte Bonferroni sonrasi anlamsiz; trend overlay'i plasebodan farksiz.
Karar: hicbiri kagit/testnete eklenmez. Canli sistem degismez (trend cekirdegi + ICT gozlem).

## 2026-09-13 ICT kazanan islem ozellik madenciligi -> RED (ict_ozellik)
Soru: kazanan ICT islemlerinin giris oncesi ozellikleri kaybedenlerden ayrilip filtreyle duzeltilebilir mi?
TRAIN (2024-09..2025-06, 98 coin, 5 dilim): ICT 1010 islem, winsorize net -0.465R; ayni barlarda rastgele yon -0.32R.
30 ozellik, lojistik + agac + basit kurallar, 4 zaman kati, 20+20 plasebo seed, 100 permutasyon: 0/10 filtre aday.
En iyi filtre kazanci +0.023R (plasebo %95: +0.063 / +0.150; permutasyon p=0.38). Tum TRAIN'e geriye bakinca her
ozellikte "iyi dilim" cikiyor, kat disinda hicbiri 4 katin 2'sinde bile pozitif degil. Kazananlarin "ortak ozellikleri"
(d <= 0.25) plasebo kazananlarinda da ayni buyuklukte. KARAR: ICT ailesi kapali; demo hesap sadece trend,
ICT arka planda bilgi amacli.

