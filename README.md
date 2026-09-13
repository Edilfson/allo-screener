# allo-screener

Kripto strateji arastirma ve kagit/testnet takip sistemi. GitHub Actions ile calisir, gercek para kullanmaz.
**Yatirim tavsiyesi degildir.** Amac bir hipotezi disiplinli olcmektir; negatif sonuc da sonuctur.

## Bilesenler

| Bilesen | Ne yapar | Zamanlama | Durum |
|---|---|---|---|
| `screener.py` + `ict_setup.py` | ICT/SMC kurulumlari (1h..1d) **arka planda kagit** takip (positions.json); testnete yeni emir ve anlik Telegram YOK (ICT_TESTNET/ICT_TELEGRAM=0), gunde bir ozet | her 30 dk | arka plan, bilgi amacli (lab: rastgele girişten farksiz, filtreyle kurtarilamiyor) |
| `trend_bot.py` | BTC/ETH/BNB gunluk MA50 sinyali + coin bazli kagit takip | 01:10 UTC | canli |
| `trend_portfoy.py` | Vol hedefli (%20, kaldiracsiz) uc kagit portfoy: b3, yuruyen top-3, H5 yuruyen top-5 + tavan %20 | 01:10 UTC | canli |
| `trend_testnet.py` | **DEMO HESABIN ANA SISTEMI**: b3 hedef agirliklarini futures testnette 1x esler (1000 USDT taban); kendi pozisyon defteriyle, ICT pozisyonlarina dokunmaz | 01:10 UTC | canli |
| `testnet_denetim.py` | Testnet hesabinda sahipsiz pozisyon (kapat) ve bayat emir (iptal) temizligi; 6 saatten yeni islemlere dokunmaz; Telegram raporu | 13:40 UTC + elle (Actions -> Testnet Denetim) | canli |
| `araclar/` | Yeniden oynatma, backtester (canli detect_ict ile esdeger), trend testleri | elle | - |
| `araclar/lab/` | Arastirma laboratuvari: harness, protokol, 12 ajanlik program, defter (1790+ test) | elle | - |

## Kanit zinciri (ozet)

- **ICT/SMC**: 98 coin x 2 yil, 2881 dolmus islem: net **-0.15R/islem** (GA sifiri disliyor); 19 parametre varyanti
  hicbiri saglam degil; rastgele-giris kontrolu: 1h'de ICT rastgeleden ayirt edilemiyor (p=0.7), 4h'de rastgeleden
  **kotu** (p=0.006). Karar: parametreyle kurtarilamaz. -> `RAPOR_2026-09-12.md`, `araclar/lab/LAB_RAPORU.md` bolum 2.3
- **Elenen diger aileler**: kesitsel momentum, intraday kirilim (brut+, net-), rejim/MR/sikisma, gunluk kirilim
  (eslesmis plasebo: kazanc tetikten degil overlay'den). -> `LAB_RAPORU.md` bolum 2
- **Hayatta kalan**: buyuk likit coinlerde gunluk **MA50 long-only + vol hedefleme**.
  - 2018-2025 train/valid: A2 (BTC/ETH/BNB) Sharpe 1.58/1.39; Y (yuruyen top-3) 1.35/1.54
  - **Holdout 2026 (tek sefer, on kayitli)**: A2 +13.5%, maxDD 14.7%, Sharpe 0.94 vs al-tut -16%/45%/-0.27; Y 1.10
  - **2011-2017 Bitstamp (Binance oncesi, bagimsiz)**: maxDD azaltma p=0.000, Sharpe farki p=0.015, mutlak getiri al-tutun altinda
  - Kirmizi takim: kod temiz; maxDD avantaji buyuk olcude pozisyon boyutu, gercek kenar Sharpe farki
  - Dogru cerceve: **dusus keser, Sharpe artirir; boga yillarinda al-tutun gerisinde kalir; kotu senaryoda 2-3 yil su alti.**

## Protokol

`araclar/lab/PROTOKOL.md`: TRAIN <2025 / VALID 2025 (tek sefer, aile basina 3 aday) / HOLDOUT 2026 (tek sefer, kapali).
Islem bazli her aile: eslesmis plasebo + 3R-winsorize + rastgele-giris kontrolu zorunlu. Her test `defter.jsonl`'a yazilir.
Yeni bulgular once `HIPOTEZLER.md`'ye (H3 trend, H4 holdout sonucu, H5 tavan/genislik) ve karar esikleri onceden yazilir.

## Calistirma

```bash
pip install -r requirements.txt
python screener.py                 # ICT taramasi (env: TELEGRAM_*, TESTNET_*)
TREND_SYMBOL=BTCUSDT,ETHUSDT,BNBUSDT python trend_bot.py   # trend + portfoy (+ TREND_TESTNET=1 ile testnet)
python trend_testnet.py --kuru     # testnet eslemesini yazmadan gor
python -m unittest araclar/test_trend_testnet.py
```

Lab betikleri `araclar/lab/` altinda; veri yollari `harness.py` icinde (yerel scratchpad). Holdout verisi kapalidir.

## Yapilmamasi gerekenler

- Kucuk orneklem kirilimina bakip parametre degistirmek
- Backtesti iyi gosterecek varsayim eklemek (%100 dolma, ileriye bakma, maliyet dusurme)
- Holdout 2026 verisini yeniden kullanmak
- Gercek paraya gecmek: kanit "dusus azaltma" icin guclu, "kar" icin degil; kagit/testnet 12 ay izlenmeli
