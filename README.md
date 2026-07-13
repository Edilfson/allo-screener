# Allo/Binance Futures EMA Pullback Screener

Binance Futures'taki tum USDT-M perpetual coinleri 4h ve 1D grafikte tarar.
Kural: swing low -> swing high arasinda **%50+ yukselis** olmus VE fiyat zirveden
geri cekilip **EMA55 veya EMA99'a temas etmis/yaklasmis** ise Telegram'a alarm atar.

## 1) Telegram bot olustur (5 dakika)
1. Telegram'da **@BotFather**'i ac, `/newbot` yaz, isim ver.
2. Sana bir **token** verecek (ornek: `123456:ABC-DEF...`). Bunu kaydet.
3. Botunla bir kere `/start` yazarak konusmayi baslat (yoksa mesaj gonderemez).
4. Chat ID'ni ogrenmek icin tarayicida su adresi ac (TOKEN yerine kendi tokenini yaz):
   `https://api.telegram.org/botTOKEN/getUpdates`
   Donen JSON icinde `"chat":{"id": 123456789 ...}` seklinde chat id'ni goreceksin.
   (Bulamiyorsan botuna bir mesaj at, sonra linki tekrar ac.)

## 2) GitHub'a yukle (ucretsiz)
1. github.com'da yeni, **public veya private** bir repo ac (orn: `allo-screener`).
2. Bu klasordeki tum dosyalari (screener.py, requirements.txt, .github/ klasoru,
   README.md) o repoya yukle. En kolay yol: GitHub web arayuzunde
   "Add file > Upload files" ile suruklayip birak.

## 3) Secrets ekle
Repo icinde: **Settings > Secrets and variables > Actions > New repository secret**
- `TELEGRAM_BOT_TOKEN` -> botfather'dan aldigin token
- `TELEGRAM_CHAT_ID` -> az once bulduğun chat id

## 4) Calistir
- **Actions** sekmesine git, "Allo Screener" workflow'unu bul.
- Otomatik olarak her 4 saatte bir calisacak (cron ayari .github/workflows/screener.yml icinde).
- Hemen test etmek istersen: Actions > Allo Screener > "Run workflow" butonuna bas.

## Ayarları degistirmek istersen (screener.py basinda):
- `RALLY_MIN_PCT = 0.50` -> minimum yukselis yuzdesi
- `TOUCH_TOLERANCE_PCT = 0.015` -> EMA'ya ne kadar yakinlasinca "temas" sayilsin
- `PULLBACK_MIN_PCT = 0.05` -> zirveden en az ne kadar geri cekilmis olmali
- `INTERVALS = ["4h", "1d"]` -> hangi zaman dilimleri taransin
- `DEDUP_COOLDOWN_HOURS = 20` -> ayni sinyal icin tekrar alarm atmadan once bekleme suresi

## Onemli notlar
- Bu tamamen ucretsiz calisir: GitHub Actions'in ucretsiz plani ayda 2000 dakika
  calisma suresi verir, bu screener'in tuketimi bunun cok altinda kalir.
- Binance API key gerekmiyor (public veri).
- Bu bir yatirim tavsiyesi degildir, sadece tarama/otomasyon aracidir; sinyal
  gelince kendi analizini yapmalisin.
