# LAB RAPORU - 2026-09-13

Kapsam: `araclar/lab/` altindaki tum ajan sonuclari (sonuc_*.json), `PROTOKOL.md`, `AJAN_SABLON.md`, `HIPOTEZLER.md`, `RAPOR_2026-09-12.md`,
`defter.jsonl`. Sadece train + valid (gelistirme) bolumleri. HOLDOUT 2026'ya hicbir ajan dokunmadi; bolum 5 ana ajan tarafindan doldurulacak.
Butun sayilar dosyalardan alinmistir; emin olunmayan yerler `(?)` ile isaretlidir.

## 1. AMAC VE PROTOKOL

**Amac:** backtestte iyi gorunen degil, YENI veride de calisan, maliyet sonrasi pozitif bir sistem. Uydurma sonuc uretmek basarisizliktir.

**Bolumler.** Gunluk (1d) kaynak: TRAIN < 2025-01-01 (2467 gun) | VALID 2025 (363 gun) | HOLDOUT 2026-01-01..bugun.
1h kaynak (veri 2024-09'da baslar): TRAIN < 2025-07-01 | VALID 2025-07..12 | HOLDOUT ayni.
VALID: her aile icin en fazla **3 aday, TEK SEFER**. HOLDOUT: sadece ana ajan, program sonunda, TEK SEFER; `harness.py` holdout verisini fiziksel
keser, ajanlar `LAB_HOLDOUT` ayarlamaz ve ham `.npy` okumaz. Veri/simulasyon sadece harness uzerinden; ileriye bakma yok (sinyal t kapanisinda,
islem en erken t+1); maliyet harness varsayilani (portfoyde 7bp/birim ciro) - maliyeti dusurerek "iyilestirme" yasak. Her konfigurasyon
`kaydet(...)` ile deftere yazilir (basarisizlar dahil).

**Aday olma sarti (kural 5, iki guncelleme ile).** Islem bazli: (a) train ort_net>0 ve GA99 alt siniri>0, valid ort_net>0 ve n>=100; (b) komsu
parametreler (+/-1 adim) de train'de pozitif; (c) **PLASEBO** - ayni stop/cikis mekanigiyle eslesmis rastgele giris kosulur, 3R-winsorize
ortalama net R plasebonunkini train VE valid'de gecmeli; (d) karin en iyi %1 islemden gelen payi < %50; (e) **rastgele-giris kontrolu** her yeni
ailenin ZORUNLU ilk adimi - bootstrap ile anlamli fark yoksa (p>0.05) aile kapanir, hedef brut kenar > 0.15R. Portfoy bazli: train VE valid'de
Sharpe ve maxDD kiyaslardan (al-tut / BTC al-tut) iyi olmali. Bu guncellemeler zorunluydu: gunluk long-only testte **rastgele giris bile train'de
+0.31R** veriyor (hayatta-kalma yanliligi + iz suren cikisin sag-carpik kuyrugu).

**Coklu test sayaci.** `defter.jsonl`: **1790 satir**, benzersiz (aile, ad, params) **949**.

| Aile | kayit | benzersiz | bolum dagilimi |
|---|---|---|---|
| tsmom2 | 513 | 69 | train 270 / valid 207 / gelistirme 36 |
| kirilim_gunluk | 261 | 146 | train 243 / valid 18 |
| eski_donem | 217 | 199 | eski_2011_2017 217 |
| saglamlik | 184 | 72 | gelistirme 172 / train 12 |
| kirilim | 151 | 93 | train 148 / valid 3 |
| tsmom | 148 | 119 | train 139 / valid 9 |
| kesitsel | 120 | 108 | train 112 / valid 8 |
| rejim | 105 | 95 | train 100 / valid 5 |
| gecikme | 68 | 27 | gelistirme 28 / train 20 / valid 20 |
| ict2 | 23 | 21 | train 19 / gelistirme 2 / valid 2 |
| **TOPLAM** | **1790** | **949** | |

Deflated Sharpe icin: tsmom + tsmom2'de TRAIN raporu olan benzersiz konfig 188, 21'i kiyas -> **N_test = 167**; bu 167 denemenin train Sharpe'i
ort 1.269 / std 0.261 / min 0.35 / maks 1.60; sirf sansla beklenen en iyi Sharpe **SR0 = 0.706 yillik**.

## 2. ELENEN AILELER (bes aile VALID'de dustu; hicbiri HOLDOUT'a goturulmedi)

### 2.1 Kesitsel momentum (108 konfig)

| Aday | TRAIN Sharpe / maxDD / CAGR | VALID Sharpe / maxDD / CAGR |
|---|---|---|
| H1_K14_U30_L20_REJVOL | 1.06 / 0.595 / +48.7% | -0.32 / 0.384 / -23.9% |
| H1_K14_U50_L10_REJVOL | 1.13 / 0.676 / +64.2% | -0.35 / 0.447 / -30.6% |
| H1_K30_U50_L20_REJVOL | 0.95 / 0.667 / +43.4% | -0.92 / 0.490 / -45.8% |
| KIYAS BTC al-tut | 0.95 / 0.832 / +51.7% | 0.05 / 0.320 / -6.5% |

Neden dustu: (1) uc adayin da VALID Sharpe'i negatif ve mutlak olarak zararda; (2) uc adayin da VALID maxDD'si (0.38/0.45/0.49) BTC'nin
0.32'sinden kotu - sadece Sharpe degil maxDD sarti da kaciriliyor; (3) alfa tek doneme sikismis (2020 +202%, 2021 +612%, 2023 +189%; 2019 -42%,
2022 -44%, 2024 -6%, 2025 -24%), 2024'ten beri BTC her yil onde. Ne ogrendik: BTC MA50 rejim filtresi 24 hucrenin 24'unde Sharpe +~0.35 / maxDD
-~0.20 sagliyor (tasinabilir bilesen). Long-short kripto kesitinde calismiyor (8/8 konfig train'de negatif, Sharpe -0.25..-0.73). K=60/90 her
yerde K=14'ten kotu - klasik 3-12 ay momentumu yok, sinyal ~2 haftalik. 2025 altcoin kesiti felaket: evren esit-agirlik al-tut -63..-78%.

### 2.2 Intraday kirilim - 1h/4h Donchian (93 konfig)

| Aday | TRAIN n / ort_net / GA99 | VALID n / ort_net / GA99 |
|---|---|---|
| donchian_1h_N55_M27_k2.0_tp5R_short | 5053 / +0.214R / [0.139, 0.287] | 4015 / -0.014R / [-0.079, 0.055] |
| donchian_4h_N20_M10_k2.0_long_rej | 2130 / +0.239R / [0.097, 0.432] | 1007 / -0.022R / [-0.170, 0.175] |
| donchian_1h_N80_M40_k1.5_long_rej | 2752 / +0.341R / [0.057, 0.709] | 1316 / -0.038R / [-0.282, 0.252] |

Neden dustu: (1) uc adayin da VALID net'i negatif iken brut'u POZITIF (+0.014/+0.044/+0.053R) - kaybeden sey maliyet; (2) islem basi maliyet
0.031-0.091R, cunku 1h/4h'te stop mesafesi %1.5-2; (3) TRAIN penceresi (2024-09..2025-06) tek rejim - 93 konfigin 85'i pozitif, 53'unun GA99 alt
siniri>0; long konfiglerinde yari1 >> yari2 (4h N40 long_rej 0.877 -> -0.091). Ne ogrendik: belirleyici kisit MALIYET, sinyal kalitesi degil ->
"stop mesafesini buyut" hipotezi dogdu (2.5). BTC MA50 overlay'i TRAIN long'u 2-3 katina cikariyor ama VALID'de tersine donuyor - yeni bilgi
eklemiyor, 2024 sonu bogasini seciyor. N>=40 gerekli, cikis kanali M~N/2; sabit TP vs trend cikisi net kazanan uretmedi (5R > TP yok > 3R,
monotonik degil = gurultu).

### 2.3 ICT2 - yapisal varyantlar + RASTGELE KONTROL (19 konfig)

Kontrol: ayni stop/TP/limit/pending/cooldown mekanigi, giris OB yerine son barin %50'si. (a) rastgele bar+yon, (b) rastgele bar + BTC MA50 yonu,
(c) ICT'nin AYNI barlari + rastgele yon. Her biri 20 seed x 98 coin. **Rastgele kontrol p degerleri (ICT eksi kontrol):**

| Dilim / bolum | ICT ort | C1 rastgele bar+yon | C1b rastgele bar, HTF yon | C1c ayni barlar, rastgele yon |
|---|---|---|---|---|
| 4h / train | -0.416R | -0.327, **p=0.0555** | -0.263, p=0.1155 | -0.498, **p=0.0055** |
| 4h / gelistirme | -0.335R | -0.267, **p=0.0425** | -0.219, p=0.083 | -0.289, **p=0.035** |
| 1h / train | -0.061R | +0.075, p=0.3655 | +0.084, p=0.3225 | -0.031, p=0.713 |
| 1h / gelistirme | -0.137R | +0.012, p=0.8435 | -0.012, p=0.8695 | -0.013, p=0.8325 |

En iyi 3 (train n / ort_net -> valid n / ort_net): **1h C2_htf_btc** 417 / +0.046R (GA95 sifiri iceriyor) -> 244 / -0.152R;
**1h C4_htf_ikisi** 284 / +0.095R (GA95 sifiri iceriyor) -> 179 / -0.086R; **4h C8_sweep_3R** 6089 / -0.075R (GA99 ust -0.017),
VALID'e goturulmedi.

Neden dustu: (1) 1h'te ICT ile rastgele giris arasinda fark YOK (p=0.37/0.84, zaman-eslesmiste 0.71/0.83) - bilgi icerigi sifir; (2) 4h'te ICT
rastgeleden ANLAMLI KOTU (zaman-eslesmis train fark -0.50R p=0.0055) - yon secimi negatif bilgi tasiyor; (3) VALID'e goturulen iki aday da
negatif -> 0 aday. Ne ogrendik: yapisal degisiklikler (OB->FVG, TP 5R->likidite swingi, limit->piyasa, HTF filtresi) kaybi kapatmiyor - 17
konfigin 15'i negatif. Likidite suprusu buyuk ornekle (n=6k-15k) test edildi: brut R ~0 (4h 3R +0.005R), net -0.08..-0.18R -> kenar tam olarak
maliyet kadar. 1h C2/C4'un train pozitifligi tamamen 2024 Q4 bogasi (2024 +0.37R / 2025 -0.15R).

### 2.4 Rejim / mean-reversion / sikisma (105 konfig)

| Aday | TRAIN ort_net / wins3R | VALID ort_net / wins3R |
|---|---|---|
| A_sqz4h_bw0.2_n20_k2.0_ls | +0.135R / +0.115 (n=1097) | -0.032R / -0.048 (n=907) |
| B_sqz4h_..._shortonly | +0.247R / +0.244 (n=530) | -0.084R / -0.086 (n=446) |
| C_brk1d_n20_k3.0_longonly | +0.503R / +0.167 (n=4457) | -0.156R / -0.214 (n=1119) |
| PLASEBO 4h (rastgele giris) | +0.698R / +0.060 (n=1665) | -0.221R / -0.269 (n=1490) |
| PLASEBO 1d (rastgele giris) | +0.308R / -0.170 (n=4838) | -0.394R / -0.529 (n=1825) |

Neden dustu: (1) uc aday da VALID'de negatif, VALID aday sayisi 0; (2) ham ortalama R yaniltici - 1d kirilimda medyan -0.28R, karin %43'u en iyi
%1 islemden, maxR=90, ve plasebo train'de +0.31R; (3) portfoy seviyesinde hicbir vol rejimi dilimi BTC al-tut Sharpe'ini (0.95) gecmiyor (orta
vol 0.74, yuksek 0.62, dusuk -0.08). Ne ogrendik: HARNESS TEMIZ - kasitli negatif kontrol ailesi (RSI2, Bollinger; 1h/4h; long ve long-short) 8/8
net negatif, GA99 UST siniri bile sifirin altinda. 3R-winsorize ortalama iyi ayirt edici (sahte kenarda ~0/negatif, gercek kenarda ham
ortalamayla ayni). Kripto trendi DUSUK vol rejiminde CALISMIYOR (Sharpe -0.08), ORTA (0.74) ve YUKSEK (0.62) volde calisiyor - islem bazli olcum
bunun TERSINI soyluyordu. 199 testlik saat/gun mevsimsellik taramasinda Bonferroni'yi gecen tek hucre yok (en buyuk t 2.55, kritik 3.83).

### 2.5 Gunluk kirilim + ESLESMIS PLASEBO (146 konfig)

Onceki ailenin onerisi test edildi: ayni Donchian, 1d dilim, stop 3*ATR14 (maliyet 0.0065R'ye dustu).

| Aday | TRAIN ort_net / wins3 / plasebo wins3 (kenar) | VALID ort_net / wins3 / plasebo wins3 (kenar) |
|---|---|---|
| d_N40_M20_k3.0_long_tp5R | +0.325R / +0.057 / -0.022 (**+0.079**) | -0.285R / -0.389 / -0.228 (**-0.161**) |
| ..._btc100_ma100 (overlay) | +0.365R / +0.079 / +0.064 (**+0.015**) | -0.328R / -0.425 / -0.370 (**-0.055**) |
| PORTFOY (risk parite %1) | Sharpe 1.04 / maxDD 0.592 / CAGR +35.3% | Sharpe -0.77 / maxDD 0.347 / CAGR -30.7% |
| KIYAS BTC al-tut | Sharpe 0.95 / maxDD 0.832 / CAGR +52.1% | Sharpe 0.05 / maxDD 0.320 / CAGR -6.5% |

Neden dustu: (1) VALID'de hem mutlak hem plaseboya gore negatif - VALID brut ortalama da negatif (-0.278R), yani 2025'te kaybeden sey maliyet
degil SINYALIN KENDISI; (2) overlay'li versiyonda TRAIN plasebo kenari sadece +0.015R - kazancin tamami "BTC MA100 ustundeyken kendi MA100'unun
ustundeki coinde long ol" kuralindan, Donchian tetiginden degil; (3) portfoy sarti saglanmiyor (TRAIN'de BTC'yi yeniyor 1.04 vs 0.95, VALID'de
yenilmiyor -0.77 vs +0.05). Ne ogrendik: maliyet hipotezi dogrulandi ama yetmedi. TRAIN'in genis pozitif bolgesi (146 konfigin ~%95'i pozitif)
DONEM ETKISI: portfoy 2021 +208..+782%, 2022 -31..-59%, 2024 -10..+5%. TP=5R olcum guvenilirligini duzeltiyor (TP'siz long'da ort_net'in
%50-100'u en iyi %1 islemden; TP=5R ile %13-16). Kural "altcoin tutmaktan iyi" ama "BTC tutmaktan kotu" (evren al-tut 2025 -82%).

## 3. HAYATTA KALAN: ZAMAN SERISI MOMENTUMU (tsmom)

Kural: gunluk kapanis > MA50 -> long, degilse nakit; esit agirlik; opsiyonel hedef vol (30 gun gerceklesen vol, taban %10, carpan tavani cap). Pencere 2018-04-01 sonrasi.

### 3.1 tsmom - parametre aramasi (108 konfig)

| Aday | TRAIN Sharpe / maxDD / CAGR | VALID Sharpe / maxDD / CAGR |
|---|---|---|
| A1_b3_ma50 (BTC/ETH/BNB) | 1.49 / 0.499 / +81.0% | 1.37 / 0.145 / +43.9% |
| A2_b3_ma50_vt30 (ana aday) | 1.59 / 0.275 / +36.1% | 1.39 / 0.085 / +27.0% |
| A3_b5_ma50_vt30 | 1.54 / 0.260 / +30.5% | 0.93 / 0.083 / +14.4% |
| KIYAS b3 al-tut | 1.06 / 0.772 / +65.2% | 0.33 / 0.384 / +3.7% |
| KIYAS BTC al-tut | 0.93 / 0.766 / +47.6% | 0.05 / 0.320 / -6.5% |
| KIYAS top10 al-tut | 0.51 / 0.864 / +7.5% | 0.17 / 0.451 / -8.9% |

A2 komsu bolgesi duz: hedef vol 20/30/40 = 1.59/1.59/1.56; MA 30..80 = 1.46/1.57/1.59/1.48/1.41/1.39; vol penceresi 14..90 = 1.53-1.60; cap
1.0/1.5/2.0/3.0 farksiz. Reddedilenler: coklu-MA oylamasi (1.41-1.48 < 1.49), evren genisletme (top5/10/20 = 0.76-0.95), long-short (funding=0'da
bile 0.91-0.95 vs 1.43-1.49), cikis histerezisi (notr, sadece ciroyu %40-50 kesiyor). Vol hedefleme ayni ortalama gross'a olceklendiginde Sharpe
1.46-1.47 (temelin ALTINDA) -> alfa degil RISK AYARI.

### 3.2 tsmom2 - sepet secim yanliligi + uygulama (150 konfig)

Yeni parametre aranmadi; ayni MA50(+vt30) kurali 7 sepette kosuldu.

| Sepet | ma50 train/valid | vt30 train/valid | al-tut train/valid |
|---|---|---|---|
| s1_btc | 1.27 / 0.33 | 1.47 / 0.13 | 0.93 / 0.05 |
| s2_btc_eth | 1.21 / 1.33 | 1.44 / 0.95 | 0.90 / 0.15 |
| **s3_orij_bnb (= A2)** | **1.49 / 1.37** | **1.59 / 1.39** | 1.06 / 0.33 |
| s3_2018_xrp | 1.13 / 0.71 | 1.37 / 0.79 | 0.91 / 0.15 |
| s3_ltc | 0.99 / 0.18 | 1.20 / 0.32 | 0.76 / 0.11 |
| s5_2018 (BTC/ETH/XRP/LTC/BCH) | 0.86 / 0.04 | 1.07 / 0.31 | 0.75 / 0.28 |
| **yuruyen_top3 (= Y)** | **0.88 / 1.56** | **1.19 / 1.54** | 0.64 / 0.20 |
| ozet (7 sepet) | medyan 1.13 / 0.71 | medyan 1.37 / 0.79 | |

**Yuruyen top-3 tarihcesi** (her ay 90g USDT hacmi, ileriye bakma yok; 93 ayda 14 degisim):
2018-04 BTC/ETH/LTC | 2020-02 BCH/BTC/ETH | 2020-10 BTC/ETH/YFI | 2022-12 BNB/BTC/ETH |
2023-03 BTC/ETH/YFI | 2023-07 BNB/BTC/ETH | 2023-12 BTC/ETH/YFI | 2024-03 BNB/BTC/ETH |
2024-10 BTC/ETH/SOL | 2024-11 BNB/BTC/ETH | 2024-12 BTC/ETH/SOL | 2025-01 BNB/BTC/ETH |
2025-02 BTC/ETH/SOL | 2025-05 BNB/BTC/ETH. **BNB ilk kez 2022-12'de ilk-3'e giriyor** - yani
A2'nin sepeti 2018'de SECILEMEZDI; train Sharpe'in ~0.2-0.6'si bu geriye donuk secimden geliyor.

**Yil bazinda (8 yil, 2018-2025) strateji al-tutu kac yilda geciyor?** GETIRIDE sadece 3-5/8 yil (ma50: s1 5, s2 4, s3_bnb 5, s3_xrp 5, s3_ltc 4,
s5 3, yuruyen 4; vt30: 3/4/3/4/4/4/4). maxDD'de 6-8/8 yil (vt30 icin 7 sepetin HEPSINDE 8/8; ma50'de 6-8). Kazanc TAMAMEN ayi yillarindan: b3'te
2018 al-tut -49% / ma50 -22% / vt30 -12%; 2022 -62% / -35% / -19%. Boga yillarinda geride (2020 +317% / +216% / +88%). Tek istisna 2025: +4% /
+44% / +27%. Bu bir getiri artirma degil **dusus kesme** stratejisi; Sharpe ustunlugu paydadan geliyor.

**Uygulama dayanikliligi (Sharpe train / valid):**

| Varyant | b3 vt30 | yuruyen vt30 |
|---|---|---|
| temel (gunluk, 7bp) | 1.59 / 1.39 | 1.19 / 1.54 |
| haftalik (pazartesi) | 1.39 / 1.19 | 1.16 / 1.21 |
| 1 gun sinyal gecikmesi | 1.55 / 1.09 | 1.17 / 1.33 |
| haftalik + 1 gun gecikme | 1.30 / 1.24 | 1.11 / 1.28 |
| maliyet 10bp | 1.57 / 1.36 | 1.17 / 1.51 |
| maliyet 20bp | 1.51 / 1.26 | 1.10 / 1.42 |
| 1 gun gecikme + 20bp | 1.47 / 0.97 | 1.08 / 1.21 |

Haftalik kontrol NET ZARARLI (ciro 20.8 -> 8.9 ama Sharpe -0.10..-0.35); maliyet 20bp'ye kadar dayanikli; en kotu kombinasyon bile valid'de
al-tutun (0.33) uzerinde. **Sonuc gunlugu (A2, gelistirme 2830 gun, Sharpe 1.57, CAGR %34.9):** en kotu 5 dusus %27.5 / %21.7 / %15.6 / %15.3 /
%12.8; en kotusu 2021-11-07 tepe -> 2023-01-04 dip (423 gun) + 308 gun toparlanma = **731 gun su alti**. A1: %49.9 / %48.5 / %36.1 / %28.9 /
%26.5, en kotusu 829 gun. Al-tut: %77.2 / %73.6 / %68.1 / %54.5 / %40.4. Ciro/yil A2 13.9 (8.6-21.2), A1 21.0; pozisyon acilisi/yil ~31 (3 coin).

### 3.3 Gecikme ajani (27 konfig)

BTC/ETH/BNB 1h veri, 2024-09-13..2025-12-31 (425 etkin gun, 37 giris / 38 cikis olayi), A2 kurali.

| Gecikme | train | valid | gelistirme | eslesmis yillik surukleme | p (zararli) |
|---|---|---|---|---|---|
| anlik / +0h | 1.20 | 2.16 | 1.61 | - | - |
| **+1h (ONERILEN)** | 1.21 | 2.18 | 1.62 | -0.19% (GA95 [-10.2, +9.5]) | 0.507 |
| +2h / +4h / +8h | 1.08 / 1.19 / 1.18 | 2.13 / 2.14 / 2.19 | 1.51 / 1.60 / 1.61 | -1.54% / -0.50% / +0.93% | 0.60 / 0.51 / 0.47 |
| +12h | 1.25 | 2.36 | 1.71 | +2.45% | 0.428 |
| **+24h (KACIN)** | 1.00 | 1.83 | 1.36 | -3.94% | 0.570 |
| al-tut kiyas | 0.95 | 0.65 | 0.84 | - | - |

Olay bazli (75 olay) yillik etki: +1h -0.43% (GA95 [-1.53, +0.66]), +2h -1.99%, +24h -4.33%. Kapanistan sonraki ILK 12 SAAT pratikte bedava;
kritik esik "ilk saatler" degil "ayni gun icinde". SAATLIK yeniden degerlendirme ZARARLI: ayni +1h'te 1.62 -> 1.38, ciro 15.7 -> 62/yil (yillik
maliyet %1.1 -> %4.3); ayrisim 0.17 Sharpe cirodan, 0.07 zamanlama gurultusunden. Gunluk kapanis bir gecikme degil bir FILTRE. Ileriye bakma
kontrolu (-1h): Sharpe 3.43 -> harness temiz.

### 3.4 Saglamlik ajani (72 kosu, sadece gelistirme)

**T1 eslesmis blok bootstrap** (blok 30 gun, B=2000, n=2830; ayni bloklar hem stratejiye hem KENDI sepetinin al-tutuna):

| Aday | Sharpe farki | GA95 | GA99 | p | maxDD farki | GA99 |
|---|---|---|---|---|---|---|
| A2 | +0.58 | [+0.064, +1.19] | [-0.122, +1.327] | 0.018 | -0.497 | [-0.759, -0.269] |
| Y | +0.64 | [+0.116, +1.211] | [-0.03, +1.384] | 0.009 | -0.560 | [-0.780, -0.347] |

"Dususu azaltiyor" iddiasi %99'da ayakta (2000 tekrarin 2000'inde ayni yon); "Sharpe'i artiriyor" iddiasi %95'te ayakta, %99'da degil.

**T2 yil bazinda MA kararliligi.** Yilin en iyi MA'si zipliyor (A2: 140/90/30/40/20/50/110/50, std 40 gun; Y std 55) ama bu KIRILGANLIK degil
DUZLUK isareti: 2021-2025 Sharpe - oracle (ileriye bakan) A2 1.59 / Y 1.29; sabit MA50 A2 1.48 / Y 1.08; walk-forward (onceki 3 yilin en iyisi)
A2 1.05 / Y 0.66; al-tut A2 0.98 / Y 0.55. Adaptif MA secimi acikca ZARARLI. Tum donemde oracle 1.83 vs sabit 1.57 -> 8 serbest secimin
asiri-uydurma primi ~0.26 Sharpe.

**T3 Monte Carlo** (93 takvim-ayi blogu, 10000 permutasyon):

| Aday | gozlenen maxDD (yuzdelik) | maxDD p95 / p99 | gozlenen su alti (yuzdelik) | su alti p95 / p99 |
|---|---|---|---|---|
| A2 | %27.5 (%75) | %34.1 / %39.3 | 730 gun (%89) | 857 / 1103 gun |
| Y | %27.3 (%72) | %34.7 / %39.3 | 819 gun (%85) | 1056 / 1317 gun |

**T4 Deflated Sharpe.** N_test=167, sigma_SR=0.261 -> SR0=0.706 yillik. A2 train (SR 1.592, carpiklik +0.60, basiklik 13.4) **DSR 0.9902**; Y
train (SR 1.192, carpiklik +1.31, basiklik 22.5) **DSR 0.9038**; gelistirme A2 0.9925 / Y 0.9348. N=50 -> 0.996/0.946; N=400 -> 0.984/0.866.
DSR>=0.95 icin gereken yillik Sharpe (N=167) 1.33. UYARI: DSR sepet secim yanliligini ve survivorship'i duzeltmez - A2'ye 0.4 indirim uygulanirsa
DSR ~0.90, yani Y ile ayni yer.

**T5 alt donemler** (Sharpe / maxDD; vt30 - duz MA50 - al-tut):

| Donem | A2 vt30 | A2 ma50 | A2 al-tut | Y vt30 | Y ma50 | Y al-tut |
|---|---|---|---|---|---|---|
| 2018-2020 | 1.73 / %21.7 | 1.49 / %48.5 | 1.01 / %77.2 | 1.50 / %20.4 | 1.16 / %47.8 | 0.67 / %83.3 |
| 2021-2022 | 0.99 / %27.4 | 1.47 / %49.8 | 0.89 / %73.6 | 0.25 / %27.1 | 0.42 / %50.1 | 0.31 / %82.2 |
| 2023-2025 | 1.78 / %15.3 | 1.54 / %28.9 | 1.19 / %40.4 | 1.56 / %16.3 | 1.23 / %30.4 | 0.88 / %52.3 |

A2 Sharpe'ta 3/3 ve maxDD'de 3/3 ustun; Y Sharpe'ta 2/3. ONEMLI: en zor donemde (2021-2022) vol hedefleme ZARARLI (A2 duz 1.47 vs vt30 0.99) ->
"vt30 > ma50" donem-dayanikli DEGIL; "MA50 > al-tut" 3/3 donemde ve iki sepette de dayanikli.

**T6 kaldirac / spot.** Strateji zaten kaldiracsiz (maks gross 1.00, ortalama 0.27 A2 / 0.24 Y). cap 1.5 -> 1.0: A2 1.57 -> 1.54. Hedef vol %20
(maks gross 0.82): A2 Sharpe 1.56 / CAGR %22.6 / maxDD %19.0 / ciro 13.9 -> 9.3; Y 1.23 / %16.6 / %18.9. Vol hedefi saf bir OLCEK dugmesi.

### 3.5 Kirmizi takim (21 bulgu, 6 denetim basligi)

Hukum: kod duzeyinde adaylar AYAKTA. 6 basligin 4'u tamamen temiz. Iki gercek indirim var, ikisi de KOD DEGIL METODOLOJI.

| Aday | onceki train S / maxDD | **duzeltilmis** train | onceki valid S / maxDD | **duzeltilmis** valid |
|---|---|---|---|---|
| A2 | 1.59 / 0.275 | **1.57 / 0.276** | 1.39 / 0.085 | **1.31 / 0.085** |
| Y | 1.19 / 0.273 | **1.36 / 0.273** | 1.54 / 0.083 | **1.47 / 0.080** |
| A2 gross-esit al-tut | - | 1.07 / 0.308 | - | 0.32 / 0.135 |
| Y gross-esit al-tut | - | 0.75 / 0.311 | - | 0.20 / 0.173 |

Not: Kirmizi takimin tablosu cap=1.0 + tum duzeltmelerle kosulmustur. Harness'e alinan iki duzeltme
(nedensel min_gun, O[t+2] maskesi) ile cap=1.5'te yeniden kosulan resmi degerler: A2 1.58/1.39 (maxDD %27.5/%8.5),
Y 1.35/1.54 (%27.3/%8.3) - holdout tablosu (bolum 5) bu kodla uretildi. Fark cap secimi ve yuvarlamadan.

**Gercek indirimler**
- **6B (YUKSEK) - maxDD avantaji boyut etkisi:** "maxDD al-tutun yarisi" iddiasinin buyuk kismi POZISYON BUYUKLUGU farkindan. A2 train: strateji
  maxDD 0.275 (ort gross 0.261) vs ham al-tut 0.772 (gross 1.0), ama **ayni ortalama gross'a olceklenmis al-tut maxDD 0.296**; Y: 0.273 vs 0.317.
  Avantaj ~0.03 -> ihmal edilebilir. Gercek katki Sharpe'ta: gross-esit kiyasa gore A2 train +0.50 / valid +0.99. (Model nakde %0 faiz yaziyor ->
  strateji ALEYHINE, muhafazakar.)
- **1E (ORTA) - min_gun sizintisi DUZELTILDI:** `harness.gunluk_panel(min_gun=200)` bir coini panele almak icin TUM veri setindeki toplam gun
  sayisina bakiyordu (ileride 200 gune ulasacagini biliyor). Sadece Y'yi etkiler; Y sepeti 174 gun boyunca secim aninda 200 gunden genc bir coin
  tutmus (agirligin %2.0'i). Nedensel uygulandiginda **Y train 1.19 -> 1.36-1.38** (maxDD ve valid degismiyor) -> sizinti ALEYHTE calisiyormus.
- **3A (ORTA) - survivorship (Y):** havuz bugunun 372 vadeli coini; yuruyen top-3, 93 ayda sadece 7 farkli coin secmis (BCH, BNB, BTC, ETH, LTC,
  SOL, YFI) - delist olmus tek sembol yok. Hayalet-coin MC (60 tekrar): 1 olay/8yil -> train 1.124; 2 olay -> 1.084 (maxDD 0.299); 5 olay ->
  0.972; 9 olay -> 0.738 (maxDD 0.378).
- **3B (ORTA) - sepet secimi (A2):** b3 = 7 sepetin en iyisi; train Sharpe'in ~0.2-0.4'u secimden.

**Temiz maddeler**
- **1B shift sondasi:** A2 shift=-1 (yarinin sinyali) train 3.86 vs 1.59; shift=+1 1.55. Gelecege kaydirmak IYILESTIRIYOR -> kod t+1 bilgisini
  kullanmiyor. **1C kesme testi:** veri 3 ayri gunde (2020-11-29 / 2023-02-07 / 2024-09-29) fiziksel kesilip zincir sifirdan hesaplandi; Cf / dv
  / vol30 / frac_ma50 / W_A2 farki tam **0.0**.
- **1D (DUSUK):** `sim_portfoy`'daki `O[t+2]` nan maskesi teknik olarak ileriye bakma ama etkilenen hucre A2 0 / Y 0 -> etki SIFIR (baska evrende
  sessizce sisirir). **1F:** 1h veriyle 0-18 saat gecikme neredeyse etkisiz (0h 1.62 / 1h 1.63 / 4h 1.62 / 12h 1.74 / 18h 1.58 / 24h 1.34) ->
  tsmom2'nin "gecikmeye cok duyarli" teshisi YANLIS.
- **2A/2B/2C:** ffill sahte 0-getiri gunu pozisyonlu hucrelerde 0; listeleme oncesi nan agirligi bozmuyor; getiri kirpmasi (-0.95, +5.0) hicbir
  hucreye binmiyor (A2 0 / Y 0). **4A/4B/4C:** gross hicbir gun 1.0'i asmiyor; cap 1.0'in etkisi -0.02 Sharpe; vol t gununde bilinen degerle
  kullaniliyor.
- **5A/5B:** maliyet formulu elle dogrulandi (max fark 0.0); train CAGR'i sifira indiren tek-yon maliyet A2 **230bp** / Y **166bp** (varsayilan
  7bp); 50bp'de bile valid A2 0.96 / Y 1.14.
- **5C / 6C (DUSUK, latent):** `tsmom2.akis()` ciroyu suruklenmemis w_prev ile hesapliyor (raporlanan ciro/yil abartili, maliyet dogru);
  `tsmom2._AY` onbellek anahtari MA listesini icermiyor (bugun etkisiz, ileride sessiz yanlis sonuc uretebilir).

### 3.6 Ek kanit: Binance oncesi donem (eski_donem, 199 benzersiz defter kaydi)

BTC gunluk MA trend takibi, Bitstamp 2011-08-18..2017-08-31 (1989 gun, 200 gun isinma sonrasi), Coinbase capraz kontrol (774 ortusen gun, kapanis
farki medyan %0.33). Bu donem hicbir parametre secimine girmedi -> GERCEK out-of-sample. Kaynak: `sonuc_eski_donem.json`
(rapor taslagi yazilirken henuz yoktu; asagidaki sayilar o dosyayla teyit edildi: MA50+vt30 Sharpe 2.59 vs al-tut 1.85,
maxDD %20.6 vs %84.8; bootstrap dSharpe +0.74 p=0.015, dmaxDD -0.64 p=0.000; 19/19 MA komsusu al-tutu geciyor;
mutlak CAGR hicbir varyantta al-tutu gecmedi: 2012-2017 icin "dusus keser, getiri artirmaz").

| Varyant (maliyet 20bp tek yon) | Sharpe | maxDD | CAGR | islem |
|---|---|---|---|---|
| BTC al-tut | 1.85 | 0.848 | +243% | 1 |
| MA20 / MA50 / MA100 / MA200 temel | 2.05 / 1.84 / 1.89 / 1.96 | 0.72 / 0.71 / 0.70 / 0.71 | +204/+185/+217/+248% | 78/49/30/15 |
| MA50+vt20 / MA50+vt30 / MA20+vt20 (cap1.0) | 2.71 / 2.59 / 2.86 | 0.140 / 0.206 / 0.159 | +71 / +106 / +73% | 49/49/78 |

Blok bootstrap (strateji - al-tut): vol hedeflemesiz MA'lar Sharpe farkinda sifiri DISLAMIYOR (MA50 -0.015 p=0.51; MA200 +0.109 p=0.25); vol
hedeflemeliler %95'te disliyor (MA50+vt30 +0.743 p=0.015; MA20+vt20 +1.007 p=0.003, %99'da da). Ayi dilimleri: 2013-12..2014-12 MA50 temel -54%
ama MA50+vt30 -4%; 2014-01..2015-08 MA50 temel -10%, MA50+vt30 +2%. Yorum: bu pencere "MA trend + vol hedefleme al-tuttan daha dusuk dususle
calisir" sonucunu bagimsiz destekliyor; ancak duz MA tek basina bu donemde al-tuta Sharpe ustunlugu SAGLAMIYOR.

## 4. DURUST BEKLENTI (HOLDOUT ONCESI, ONCEDEN YAZILI)

1. **Sharpe ~1.0-1.35.** Yanlilik duzeltmeleri: sepet secimi -0.2..-0.4 (A2), survivorship -0.1..-0.45 (Y), coklu test esigi SR0=0.71. tsmom2
   7-sepet medyani train 1.37 / valid 0.79; saglamlik ajani 1.0-1.2; kirmizi takim A2 ~1.35 / Y ~0.9-1.25. Tek yilin Sharpe'i ~+/-0.9 standart
   hatayla gelir -> **2026'da 0.7-1.0 bile basaridir**. A2'nin Y'den belirgin iyi cikmasi BEKLENMEMELI (aralarindaki 0.34 fark istatistiksel
   olarak yok).
2. **maxDD avantaji kucuk.** Ham al-tut kiyasi (0.77 vs 0.28) yaniltici; BRUT-ESLENMIS kiyasa gore A2 0.276 vs 0.308, Y 0.273 vs 0.311. "Dususu
   yariya indirir" ancak "daha az yatirimda kalarak" dogru. Gercek katki Sharpe'ta (gross-esit kiyasa gore A2 train +0.50 / valid +0.99).
3. **Kotu senaryo 2-3 yil su alti.** MC %95 dilimi: A2 maxDD %34 + 857 gun (2.3 yil); Y %35 + 1056 gun (2.9 yil). Gozlenen en kotu dusus zaten
   731 (A2) / 819 (Y) gun su alti. Canli kullanimda asil risk kayip degil **terk etme**.
4. **Tek coin gap riski.** Tek-coin agirlik ort 0.19 / p95 0.37 / maks 0.47; bir coinin gecelik -%90 cokusu portfoyde ~%17 kayip. MA50 ve 30
   gunluk gecmis vole bakan hedefleme bu riske karsi HICBIR koruma saglamaz (hayalet-coin MC: 2 olay -> Y train 1.19 -> 1.08; 9 olay -> 0.74).
5. **Boga piyasasinda geride kalinacak** (8 yilda getiride sadece 3-5 yil onde; 2020 al-tut +317% vs vt30 +88%).
6. **Uygulama.** Ayni gun icinde islem yapildigi surece gecikme maliyeti olculemiyor (+1h yillik -0.4%, GA95 [-1.5%, +0.7%]); tam gun kacirmak
   -4..-6%/yil ve -0.25 Sharpe. Maliyet 20bp'ye kadar dayanikli (basabas 166-230bp).

## 5. HOLDOUT 2026 SONUCU

Calistirma: 2026-09-13 ~23:04 UTC, `LAB_HOLDOUT=son-degerlendirme-2026 python holdout_degerlendir.py`, TEK SEFER.
Donem: 2026-01-01..2026-09-12 (252 getiri gunu). Adaylar bolum 4'te ve HIPOTEZLER.md H4'te ONCEDEN ilan edilmisti;
sonradan aday eklenmedi, parametre degistirilmedi. Kaynak: `sonuc_holdout.json`.

| aday | getiri | CAGR | maxDD | Sharpe | ciro/yil | kendi al-tutu (getiri / maxDD / Sharpe) |
|---|---|---|---|---|---|---|
| A1 b3 MA50 | +11.2% | +16.6% | 18.6% | 0.69 | 21.7 | -16.0% / 45.0% / -0.27 |
| **A2 b3 MA50 + vt30** | **+13.5%** | +20.1% | **14.7%** | **0.94** | 22.6 | -16.0% / 45.0% / -0.27 |
| **Y yuruyen top-3 MA50 + vt30** | **+17.9%** | +27.0% | **14.7%** | **1.10** | 22.2 | +7.7% / 45.0% / 0.47 |
| Y yuruyen top-3 MA50 (vt yok) | +33.0% | +51.2% | 18.6% | 1.27 | 23.7 | +7.7% / 45.0% / 0.47 |
| K BTC al-tut | -13.8% | -19.4% | 39.5% | -0.23 | 0 | - |

Brut-eslenmis al-tut kiyasi (kirmizi takim 6B'nin istedigi adil DD kiyasi):

| aday | strateji Sharpe / maxDD | al-tut ayni brutte Sharpe / maxDD |
|---|---|---|
| A2 (brut 0.40) | 0.94 / 14.7% | -0.27 / 20.0% |
| Y  (brut 0.37) | 1.10 / 14.7% | 0.47 / 18.6% |

**Karar: 4/4 aday on kayitli olcutu gecti** (Sharpe VE maxDD, hem ham hem brut-eslenmis kiyastan iyi).
Sonuc, bolum 4'te onceden yazilan beklenti araliginin (Sharpe 0.7-1.35) icinde; ne beklenenden iyi ne kotu.
2026 dusen bir yildi (BTC -13.8%); strateji tam da tasarlandigi seyi yapti: dususu kesti, kucuk pozitif getiri verdi.
Tek yil oldugu icin bu bir "dogrulama", "kanit" degil; kanit zinciri 2012-2017 + 2018-2024 + 2025 + 2026 dort bagimsiz
pencerede ayni yonde olmasidir.

Bundan sonra: holdout verisi KAPALI (yeniden kullanilamaz). Yeni karar ancak canli kagit/testnet verisiyle (H3: 12 ay).

## 6. CANLI UYGULAMA

Zincir: `trend_bot.py` gunde bir kez (01:10 UTC) kosar -> icinden `trend_portfoy.py` cagrilir -> `TREND_TESTNET=1` ise `trend_testnet.py` ile testnete eslenir. Durum `trend_state.json`'da.

**trend_bot.py (sinyal katmani).** Kural: gunluk kapanis > MA50 -> YATIRIMDA, altinda -> NAKITTE; kapanmamis son mum atilir. Evren:
`TREND_SYMBOL` verilmezse hacme gore ilk `TREND_TOP` (vars. 50) USDT paritesi (kaldiracli token ve stablecoinler filtreli). Durum degisiminde
Telegram'a AL/SAT mesaji + grafik; sinyaller `signals_log.json`'a (screener ile ORTAK) yazilir. Kagit takip: coin basi equity 1.0, durum
degisince tek yon `TREND_MALIYET` (vars. %0.1) kesilir, al-tut yaninda tutulur. Veri: `data-api.binance.vision` (anahtarsiz).

**trend_portfoy.py (portfoy katmani, lab sonucu).** Agirlik `araclar/lab/tsmom.py` ile birebir:
`w_i = (1/K) * sinyal_i * min(HEDEF_VOL / vol_i, CAP)`, vol = 30 gunluk yillik gerceklesen vol
(taban %10); brut > 1 ise olceklenir -> **kaldirac yok**. Parametreler: `MA=50`, `VOL_WIN=30`,
**`HEDEF_VOL=0.20` (vt20)**, **`CAP=1.0`**, `MALIYET=0.001` (|dw| basina) - saglamlik ajaninin
spot onerisiyle ayni (Sharpe degismiyor, maks %82 yatirimli, ciro %33 dusuk). Iki sepet PARALEL:
`b3` = BTC/ETH/BNB (sabit, referans) ve `yuruyen3` = her ay basi onceki gunun 90 gunluk ort USDT
hacmine gore ilk 3 (vadeli listesinden, stablecoin/altin tokenleri haric; ay icinde sabit).
Her sepet icin kagit equity, al-tut, ciro maliyeti, zirve ve guncel dusus tutulur; Telegram
mesajinda beklenti acikca yaziyor ("Sharpe ~1.0-1.2, maxDD ~%19-28, kotu senaryo 2-3 yil su alti").
Guncel durum (2026-09-12): b3 w = BTC 0.140 / ETH 0.091 / BNB 0.145 (brut ~%37); yuruyen sepet
2026-09 = BTC/ETH/SOL, w = 0.140 / 0.091 / 0.101. Kagit takip 2026-09-11'de basladi -> henuz veri yok.

**trend_testnet.py (opsiyonel esleyici).** `state['portfoy']['b3']['w']` hedef agirliklarini
Binance **Futures testnet**'te **1x kaldiracla** pozisyona cevirir:
`hedef_notional_i = TABAN * w_i`, TABAN = `TREND_TESTNET_USDT` (vars. **1000 USDT**);
`delta = hedef - mevcut (positionRisk.positionAmt)`; `|delta * fiyat| < 25 USDT` ise emir YOK.
Emir tipi MARKET, azaltmada `reduceOnly=true`. ICT screener ile ortak hesap korumasi: hicbir acik
emri (limit/stop/TP) iptal etmez ve ayni sembolde son 24 saatte basarili ICT giris emri varsa o
sembol ATLANIR. `--kuru` modu (veya API anahtari yoksa otomatik) API'ye hicbir sey yazmaz;
kagit takip etkilenmez, testnet hatasi bot akisini bozmaz.

## 7. SONRAKI ADIMLAR VE YAPILMAMASI GEREKENLER

**Yapilacaklar**
1. **HOLDOUT 2026'yi TEK SEFER kos** (`LAB_HOLDOUT=... python holdout_degerlendir.py`). Adaylar HIPOTEZLER.md H4'te onceden ilan edildi:
   A1_b3_ma50, A2_b3_ma50_vt30 (ana), Y_yuruyen3_ma50_vt30, Y_yuruyen3_ma50; kiyaslar b3 al-tut, BTC al-tut, yuruyen top-3 al-tut. Betik
   BRUT-ESLENMIS al-tut kiyasini da basiyor (kirmizi 6B). Olcut: "Sharpe > kendi kiyasi VE maxDD < kiyasin yarisi" ikili sarti - ciplak Sharpe
   siralamasi degil.
2. **A2 ve Y'yi BIRLIKTE degerlendir:** farkin devam edip etmemesi bilgi verir (A2 iyi cikarsa sansin devami, kotu cikarsa secim yanliliginin
   bedeli).
3. **Orneklem buyut** (tek gercek istatistiksel is): ayni MA50 kuralini kripto disi varliklara (hisse endeksi, emtia, FX gunluk) uygula;
   eski_donem calismasini genislet. 8 yil / 4-6 bagimsiz rejim kisitini asmanin tek durust yolu budur.
4. **Uygulama:** hedef vol %20 + cap 1.0 (mevcut ayar), spot, kaldiracsiz; kapanistan sonra ayni gun icinde islem; "kacirilan gun" alarmi + ayni
   gun telafi kosusu (orn. 13:00 UTC yedek tetik). Saniye hassasiyeti gereksiz.
5. **Kullaniciya onceden yazili risk bildirimi:** %95 kotu senaryo 2.3-2.9 YIL su alti.
6. **Kod temizligi (kirmizi takim):** `harness.gunluk_panel` min_gun filtresini kalici olarak nedensel yap; `sim_portfoy`'daki `O[t+2]` maskesini
   kaldir; `tsmom2.akis()` ciro hesabini suruklenmis w_prev ile duzelt; `tsmom2._AY` anahtarina MA'yi ekle.
7. **Olculmemis riskler:** pozisyon boyutunun vol tahminine duyarliligi, borsa kesintisi / emir reddi, tek coin gecelik gap. (Agirlik tavani
   dusunulebilir - ama bu YENI bir parametre secimi olur: once yaz, sonra test et.)

**Yapilmamasi gerekenler**
1. **ICT/SMC'ye donmek:** parametre sorunu degil, sinyalin bilgi icerigi yok (1h p=0.37/0.84; 4h rastgeleden anlamli KOTU p=0.0055).
2. **Kapatilan aileleri yeniden acmak:** kesitsel momentum, intraday Donchian, gunluk Donchian, sikisma kirilimi, mean-reversion. Donchian uc
   dilimde de (1h/4h/1d) test edildi - maliyet bahanesi de kalmadi.
3. **tsmom ailesinde YENI PARAMETRE ARAMAK:** 108 + 150 = 258 konfig denendi; SR0 = 0.71 ve her yeni deneme esigi yukseltiyor. Komsu bolge zaten
   duz (mukemmel yillik ongoru bile sadece +0.11..+0.26 Sharpe ekliyor).
4. **Yillik / adaptif MA secimi:** walk-forward sabit MA50'nin 0.42-0.43 ALTINDA; tasinma gucu sifir, hatta negatif.
5. **Haftalik rebalans veya saatlik yeniden degerlendirme:** haftalik Sharpe -0.10..-0.35; saatlik 1.62 -> 1.38, ciro 4 kati.
6. **Evren genisletme** (top5/10/20, b5, b6): bes ayri sekilde denendi, hepsi kotu.
7. **Long-short / short bacagi:** hem kesitselde hem tsmom'da yapisal olarak zararli; funding sifir olsa bile long-only'nin alti.
8. **Maliyet varsayimini dusurerek "iyilestirme":** protokol yasakliyor, ayrica gereksiz (basabas 166-230bp vs 7bp).
9. **HOLDOUT'a sonradan aday eklemek:** H4 on kaydi kapali, aday EKLENMEZ.
10. **"maxDD al-tutun yarisi" cumlesini duzeltmeden tekrarlamak:** brut-eslenmis kiyasa gore fark ~0.03 -> yaniltici olur.

---
*Kaynaklar: araclar/lab/{PROTOKOL.md, AJAN_SABLON.md, sonuc_{kesitsel,kirilim,ict2,rejim,kirilim_gunluk,tsmom,tsmom2,gecikme,saglamlik,kirmizi}.json,
defter.jsonl, holdout_degerlendir.py}, HIPOTEZLER.md, RAPOR_2026-09-12.md, trend_{bot,portfoy,testnet}.py, trend_state.json.*

## 8. EK TUR (2026-09-13): EMA GERI CEKILME, TERSINE DONUS, FUNDING

Kullanici istegiyle ayni protokolde uc yeni aile (holdout kapali; sadece TRAIN/VALID). Hepsi RED.

| aile | konfig | TRAIN | VALID | neden dustu |
|---|---|---|---|---|
| ema_geri (4h/1d) | 76 | 0 aday | - | trend-ici rastgele plaseboyu anlamli gecemedi; 4h ikinci yari 27/27 negatif; filtre plaseboyu da yukseltiyor |
| tersine (1d kesit) | 144 | 0 aday | - | L/S maliyetsiz bile ~0; ciro 150-440x/yil; long-only = beta; hayatta-kalma yanliligi lehine olmasina ragmen negatif |
| funding carry | 23 (+12 sinyal testi) | 22/23 olcutu gecti | 0/3 | 2025'te getiri ~0 / negatif (yeni altcoinlerin negatif funding'i); 2023-25 nakit faizinin altinda |

Ogrenilen: (1) "trendde olmak" disinda giris zamanlamasi eklenen hicbir kurulum plaseboyu gecmedi (ema_geri, gunluk
kirilim, ict2); (2) altcoin kesit siralamasi iki yonde de maliyet sonrasi olu (kesitsel, tersine); (3) funding carry
gercek bir primdi ama 2021 sonrasi erimis bir rejim; kagitta dusuk oncelikli izlenebilir, getiri stratejisi degil.
Kaynak: sonuc_ema_geri.json, sonuc_tersine.json, sonuc_funding.json.
