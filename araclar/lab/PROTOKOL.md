# ARASTIRMA PROTOKOLU (2026-09-13)

Amac: gercek, tekrarlanabilir, maliyet sonrasi pozitif bir sistem bulmak. "Backtestte iyi
gorunen" degil, **yeni veride de calisan**. Uydurma sonuc uretmek basarisizliktir.

## Bolumler
- Gunluk (1d) kaynakli testler: TRAIN < 2025-01-01 | VALID 2025
- 1h kaynakli testler (1h..12h dilimler; veri 2024-09'da basliyor): TRAIN < 2025-07-01 | VALID 2025-07..12
  -> `rapor(..., kaynak="1h")` ve `bolum_filtre(..., kaynak="1h")` kullan.
- VALID: her aile icin en fazla **3** aday, TEK sefer
- HOLDOUT 2026         : SADECE ana ajan, programin sonunda, hayatta kalanlar icin TEK sefer.
  `harness.py` holdout verisini fiziksel keser. Ajanlar `LAB_HOLDOUT` degiskenini ASLA ayarlamaz,
  ham `.npy` dosyalarini dogrudan okumaz.

## Her ajanin uymasi gerekenler
1. Sadece `harness.py` fonksiyonlarini kullan (veri, sim_islem / sim_portfoy, rapor, kaydet).
2. **Her** calistirilan konfigurasyonu `kaydet(...)` ile deftere yaz (basarisizlar dahil).
3. Ileriye bakma yok: sinyal t barinin kapanisinda, islem en erken t+1'de. Gosterge hesabinda
   gelecek bar kullanma (ornek: swing teyidi k bar gecikmeli).
4. Maliyet: harness varsayilani. Maliyeti dusurerek "iyilestirme" yasak.
5. Aday olma sarti (islem bazli): train ort_net>0 ve GA99 alt siniri>0; valid ort_net>0 ve n>=100;
   komsu parametreler (±1 adim) de train'de pozitif. Portfoy bazli: train ve valid'de Sharpe ve maxDD
   kiyas (al-tut / BTC al-tut) ile karsilastirilir, ikisinde de daha iyi olmali.
6. Rapor formati (ajan ciktisi): JSON — aile, denenen_sayisi, en_iyi_3 [{ad, params, train, valid, komsular}],
   dusenler (kisa), ogrenilen (3-5 madde), oneri (bir sonraki ajan icin).
7. Ajan kodunu `araclar/lab/<aile>.py` olarak birakir; tekrar calistirilabilir olmali.

## KURAL 5 GUNCELLEMESI (rejim ajanindan, 2026-09-13): plasebo + winsorize ZORUNLU
Gunluk long-only islem testlerinde RASTGELE giris bile train'de +0.31R veriyor (data1d_all
hayatta-kalma yanliligi + iz suren cikisin sag-carpik kuyrugu: karin %97'si en iyi %1 islemden).
Bu yuzden islem bazli aday sarti:
  (a) ayni stop/cikis mekanigiyle ESLESMIS PLASEBO (rastgele giris, ayni bar orani) kos;
  (b) 3R-winsorize ortalama net R > plasebonun winsorize ortalamasi, train VE valid'de;
  (c) karin en iyi %1 islemden gelen payi < %50.
`rejim.py` icindeki `sig_plasebo()` ve `saglam()` dogrudan import edilebilir.

## Zorunlu ilk adim: rastgele-giris kontrolu (ict2 ajanindan, 2026-09-13)
Islem bazli her yeni aile once KONTROL kosar: ayni stop/TP/giris mekanigi, ayni barlar, RASTGELE yon
(veya rastgele bar). Sinyal kumesi kontrolden bootstrap ile anlamli farkli degilse (p>0.05) aile kapanir.
Hedef brut kenar > 0.15R; altinda maliyet yer (1h/4h'te islem basi maliyet 0.03-0.12R).
Kanit: ICT 1h p=0.71 (fark yok), 4h ICT rastgeleden -0.50R KOTU (p=0.006).

## Coklu test uyarisi
Defterdeki toplam test sayisi N ise, N test arasinda en iyisinin sansla GA99'u gecme
olasiligi kucuk degildir. Bu yuzden VALID tek sefer ve HOLDOUT tek sefer. Aile icinde
"en iyi parametre" degil, "genis parametre bolgesi pozitif" aranir.

## Bilinen sonuclar (tekrar deneme)
- ICT/SMC (BOS+FVG+OB, limit OB50, 5R): 2881 islem net -0.15R; 19 varyant hicbiri saglam. H1 ters-trend RED.
- Mean-reversion range LH/HL: -16.5R (eski test). TP mesafe taramasi 210 kombinasyon: 206 zarar.
- Altcoin sepeti MA trend tek basina zayif (2025 -50%+); BTC/ETH/BNB gunluk MA50 saglam (IS+OOS).
