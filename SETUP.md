# Kurulum Rehberi

Adim adim kurulum. Her sey sifirdan, hic bir sey bilmiyorsan bile kurabilirsin.

---

## Icindekiler

1. [Gereksinimler](#1-gereksinimler)
2. [Python Kurulumu](#2-python-kurulumu)
3. [Projeyi Indir](#3-projeyi-indir)
4. [Bagimliliklar](#4-bagimliliklar)
5. [Tesseract OCR](#5-tesseract-ocr-opsiyonel)
6. [Telegram Bot Olustur](#6-telegram-bot-olustur)
7. [API Anahtarlari](#7-api-anahtarlari)
8. [Bilkent Bilgileri](#8-bilkent-bilgileri)
9. [.env Dosyasini Doldur](#9-env-dosyasini-doldur)
10. [Calistir](#10-calistir)
11. [Sunucu Kurulumu (VPS)](#11-sunucu-kurulumu-vps)
12. [Sorun Giderme](#12-sorun-giderme)

---

## 1. Gereksinimler

| Gereksinim | Minimum | Tavsiye |
|-----------|---------|---------|
| Python | 3.11 | 3.12 |
| RAM | 2 GB | 4 GB |
| Disk | 2 GB | 5 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 (sunucu) |

**Hesaplar:**
- Bilkent Moodle hesabi (ogrenci)
- Bilkent STARS hesabi (ogrenci numarasi + sifre)
- Bilkent Webmail hesabi (IMAP erisimi)
- Telegram hesabi
- Google AI Studio hesabi (ucretsiz)
- OpenAI hesabi (ucretli, ayda ~$0.90)

---

## 2. Python Kurulumu

### Windows
1. [python.org/downloads](https://www.python.org/downloads/) adresinden Python 3.12 indir
2. Kurulum sirasinda **"Add Python to PATH"** kutucugunu **mutlaka** isaretle
3. Dogrula:
```bash
python --version
# Python 3.12.x
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip -y
```

### macOS
```bash
brew install python@3.12
```

---

## 3. Projeyi Indir

```bash
git clone <repo-url>
cd Moodle_Student_Tracker
```

veya ZIP olarak indir ve bir klasore cikar.

---

## 4. Bagimliliklar

### Sanal ortam olustur (tavsiye edilir)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Paketleri kur

```bash
pip install -r requirements.txt
```

> **Not:** Ilk kurulumda `sentence-transformers` embedding modelini indirecek (~100 MB). Bu bir kere olur.

> **Windows hatasi?** `faiss-cpu` veya `pymupdf` kurulumunda hata alirsan:
> ```bash
> pip install --upgrade pip setuptools wheel
> pip install -r requirements.txt
> ```

---

## 5. Tesseract OCR (Opsiyonel)

Sadece **taranmis (scanned) PDF** dosyalarin varsa gerekli. Cogu ders materyali zaten text-based PDF oldugu icin opsiyonel.

### Windows
1. [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki) adresinden `.exe` indir
2. Kurulum sirasinda **Turkish** dil paketini sec
3. Kurulum yolunu PATH'e ekle (genelde `C:\Program Files\Tesseract-OCR`)

### Ubuntu/Debian
```bash
sudo apt install tesseract-ocr tesseract-ocr-tur -y
```

### macOS
```bash
brew install tesseract tesseract-lang
```

### Dogrula
```bash
tesseract --version
# tesseract 5.x.x
```

---

## 6. Telegram Bot Olustur

1. Telegram'da **@BotFather**'i bul
2. `/newbot` komutunu gonder
3. Bot icin bir isim ver (ornek: "Moodle Asistan")
4. Bot icin bir username ver (ornek: `moodle_asistan_bot`) — `_bot` ile bitmeli
5. BotFather sana bir **token** verecek:
   ```
   7234567890:AAH1234abcdef...
   ```
   Bu token'i kaydet → `.env` dosyasina `TELEGRAM_BOT_TOKEN` olarak gireceksin.

### Telegram Chat ID'ni Bul

1. Telegram'da **@userinfobot**'u bul
2. `/start` gonder
3. Sana chat ID'ni verecek (ornek: `649226694`)
4. Bu ID'yi kaydet → `.env` dosyasina `TELEGRAM_OWNER_ID` olarak gireceksin.

---

## 7. API Anahtarlari

### Google Gemini (Ucretsiz — ana chat motoru)

1. [aistudio.google.com/apikey](https://aistudio.google.com/apikey) adresine git
2. Google hesabinla giris yap
3. "Create API Key" tikla
4. API key'i kopyala → `.env` dosyasina `GEMINI_API_KEY` olarak gir

> Ucretsiz plan: Gunluk 1500 istek, cogu kullanim icin yeterli.

### OpenAI (Ucretli — intent + extraction)

1. [platform.openai.com/api-keys](https://platform.openai.com/api-keys) adresine git
2. Hesap olustur veya giris yap
3. "Create new secret key" tikla
4. API key'i kopyala → `.env` dosyasina `OPENAI_API_KEY` olarak gir
5. Bakiye yukle: **$5 yeterli** (aylar yeter)

> Kullanilan modeller: GPT-4.1-mini (intent, ~$0.016/1K istek) ve GPT-4.1-nano (extraction, ~$0.005/1K istek)
> Tahmini aylik maliyet: **~$0.90**

### Z.ai / GLM (Opsiyonel fallback)

Zorunlu degil. Gemini ve OpenAI yeterliyse bos birakabilirsin.

---

## 8. Bilkent Bilgileri

### Moodle
- **MOODLE_URL**: Bilkent Moodle adresi. **Her donem degisir!**
  - Ornek: `https://moodle.bilkent.edu.tr/2025-2026-spring`
  - Yeni donemde bu URL'yi guncelle
- **MOODLE_USERNAME**: Bilkent kullanici adin (ogrenci numarasi veya email)
- **MOODLE_PASSWORD**: Moodle sifren

> Token otomatik alinir ve `data/.moodle_token` dosyasina kaydedilir. Manuel token girmene gerek yok.

### STARS
- **STARS_USERNAME**: Bilkent ogrenci numaran
- **STARS_PASSWORD**: STARS sifren

> Bot, STARS'a her 10 dakikada otomatik giris yapar. 2FA kodu email'den otomatik okunur.

### Webmail (IMAP)
- **WEBMAIL_EMAIL**: Bilkent email adresin (ornek: `ad.soyad@ug.bilkent.edu.tr`)
- **WEBMAIL_PASSWORD**: Email sifren

> STARS 2FA kodlarinin otomatik okunmasi icin gerekli. Ayrica AIRS/DAIS mail bildirimleri icin kullanilir.

---

## 9. .env Dosyasini Doldur

```bash
cp .env.example .env
```

Simdi `.env` dosyasini bir text editor ile ac ve doldur:

```bash
# ─── Moodle ──────────────────────────────────────────────
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=22003467
MOODLE_PASSWORD=sifreni_buraya_yaz

# ─── LLM API Keys ───────────────────────────────────────
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...

# ─── Telegram Bot ────────────────────────────────────────
TELEGRAM_BOT_TOKEN=7234567890:AAH1234...
TELEGRAM_OWNER_ID=649226694

# ─── STARS ───────────────────────────────────────────────
STARS_USERNAME=22003467
STARS_PASSWORD=stars_sifren

# ─── Webmail IMAP ────────────────────────────────────────
WEBMAIL_EMAIL=ad.soyad@ug.bilkent.edu.tr
WEBMAIL_PASSWORD=email_sifren
```

**Geri kalan ayarlar (MODEL_CHAT, EMBEDDING_MODEL, vs.) degistirme — varsayilanlar optimize edilmis durumda.**

---

## 10. Calistir

```bash
python telegram_bot.py
```

Basarili cikti:
```
🔧 Bilesenler yukleniyor...
🚀 Bot calisiyor! (Owner: 649226694)
🔄 Auto-sync: Her 10 dk | Odev check: Her 10 dk
📧 Mail: Her 30 dk | STARS auto-login: Her 10 dk
   Ctrl+C ile durdur
```

Simdi Telegram'da botunu bul ve `/start` gonder.

### Ilk calistirmada ne olur?
1. Moodle'a otomatik giris yapilir
2. Webmail'e baglanilir
3. STARS'a giris yapilir (2FA kodu email'den otomatik okunur)
4. Ders materyalleri indirilir ve indexlenir (ilk sync 2-5 dakika surebilir)
5. Hazir! Artik soru sorabilirsin.

---

## 11. Sunucu Kurulumu (VPS)

Botun 7/24 calismasi icin bir VPS (sanal sunucu) gerekli.

### Tavsiye: Hetzner / DigitalOcean / Contabo
- **Minimum:** 2 vCPU, 4 GB RAM, 40 GB SSD
- **Maliyet:** ~$4-6/ay

### Sunucuda Kurulum (Ubuntu 22.04)

```bash
# 1. Sistem guncelle
sudo apt update && sudo apt upgrade -y

# 2. Python + gerekli paketler
sudo apt install python3.12 python3.12-venv python3-pip tesseract-ocr tesseract-ocr-tur -y

# 3. Proje klasoru olustur
sudo mkdir -p /opt/moodle-bot
cd /opt/moodle-bot

# 4. Sanal ortam
python3.12 -m venv venv
source venv/bin/activate

# 5. Dosyalari kopyala (kendi bilgisayarindan)
# Kendi bilgisayarinda su komutu calistir:
# scp -r ./* root@SUNUCU_IP:/opt/moodle-bot/

# 6. Bagimliliklar
pip install -r requirements.txt

# 7. .env dosyasini olustur ve doldur
cp .env.example .env
nano .env
```

### systemd Servisi (7/24 calisma)

```bash
sudo nano /etc/systemd/system/moodle-bot.service
```

Icerik:
```ini
[Unit]
Description=Moodle Student Tracker Telegram Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/moodle-bot
ExecStart=/opt/moodle-bot/venv/bin/python telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Servisi aktiflestir ve baslat
sudo systemctl daemon-reload
sudo systemctl enable moodle-bot
sudo systemctl start moodle-bot

# Durumu kontrol et
sudo systemctl status moodle-bot

# Loglari izle
journalctl -u moodle-bot -f
```

### Guncelleme (yeni kod deploy etme)

Kendi bilgisayarindan:
```bash
# Syntax kontrol (deploy etmeden once)
python -c "import ast; ast.parse(open('telegram_bot.py').read()); print('OK')"

# Dosyalari gonder
scp telegram_bot.py main.py root@SUNUCU_IP:/opt/moodle-bot/
scp core/*.py root@SUNUCU_IP:/opt/moodle-bot/core/

# Servisi yeniden baslat
ssh root@SUNUCU_IP "systemctl restart moodle-bot"
```

---

## 12. Sorun Giderme

### "ModuleNotFoundError: No module named 'xxx'"
```bash
pip install -r requirements.txt
```
Sanal ortami aktif etmeyi unutma (`source venv/bin/activate` veya `venv\Scripts\activate`).

### "MOODLE_TOKEN invalid" veya Moodle baglanti hatasi
- `MOODLE_URL`'nin dogru doneme ait oldugundan emin ol
- Bilkent Moodle URL'si **her donem degisir** (ornek: `2025-2026-spring`)
- `data/.moodle_token` dosyasini sil ve botu yeniden baslat

### STARS giris hatasi / 2FA kodu bulunamadi
- Webmail bilgilerinin dogru oldugunu kontrol et
- IMAP erisiminin acik oldugunu dogrula (Bilkent webmail genelde varsayilan acik)
- STARS sifreni degistirdiysen `.env`'i guncelle

### "faiss" veya "sentence-transformers" kurulum hatasi
```bash
pip install --upgrade pip setuptools wheel
pip install faiss-cpu sentence-transformers
```

### Bot calisiyor ama mesajlara cevap vermiyor
- `TELEGRAM_OWNER_ID`'nin dogru oldugunu kontrol et
- Bot sadece owner'a cevap verir (guvenlik)
- Telegram'da botu `/start` ile baslat

### Indexleme cok yavas
- Normal: Ilk sync 2-5 dakika surebilir (tum materyaller indiriliyor)
- Buyuk PDF'ler (400+ sayfa) 2-3 dakika surebilir
- Sonraki sync'ler sadece yeni dosyalari kontrol eder (saniyeler icerisinde)

### Sunucuda "Killed" hatasi (bellek yetersiz)
```bash
# Swap ekle (2 GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Sifirdan indexleme (tum verileri temizle)
```bash
rm -f data/faiss.index data/metadata.json data/sync_state.json
# Botu yeniden baslat, sonra Telegram'dan /sync gonder
```

### Log kontrol
```bash
# Sunucuda
journalctl -u moodle-bot -f

# Veya direkt
python telegram_bot.py 2>&1 | tee bot.log
```
