# Kurulum Rehberi

Adim adim kurulum. Sifirdan, hicbir sey bilmesen bile kurabilirsin.

---

## Icindekiler

1. [Gereksinimler](#1-gereksinimler)
2. [Python Kurulumu](#2-python-kurulumu)
3. [Projeyi Indir](#3-projeyi-indir)
4. [Bagimliliklari Kur](#4-bagimliliklari-kur)
5. [Tesseract OCR (Opsiyonel)](#5-tesseract-ocr-opsiyonel)
6. [Telegram Bot Olustur](#6-telegram-bot-olustur)
7. [API Anahtarlari](#7-api-anahtarlari)
8. [Bilkent Moodle Bilgileri](#8-bilkent-moodle-bilgileri)
9. [.env Dosyasini Doldur](#9-env-dosyasini-doldur)
10. [Calistir](#10-calistir)
11. [Sunucu Kurulumu (VPS)](#11-sunucu-kurulumu-vps)
12. [Sorun Giderme](#12-sorun-giderme)

---

## 1. Gereksinimler

| Gereksinim | Minimum | Tavsiye |
|-----------|---------|---------|
| Python | 3.10 | 3.12 |
| RAM | 2 GB | 4 GB |
| Disk | 2 GB | 5 GB |
| OS | Ubuntu 22.04 / Windows 10 | Ubuntu 24.04 |

**Not:** Embedding modeli (paraphrase-multilingual-MiniLM-L12-v2) ilk calistirmada ~500 MB indirir.

---

## 2. Python Kurulumu

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 --version   # 3.10+
```

### Windows

[python.org](https://www.python.org/downloads/) adresinden Python 3.10+ indirip kur. Kurulum sirasinda "Add Python to PATH" secinegini isaretlemeyi unutma.

---

## 3. Projeyi Indir

```bash
git clone https://github.com/onurcangnc/Moodle_Student_Tracker.git
cd Moodle_Student_Tracker
```

---

## 4. Bagimliliklari Kur

```bash
# Sanal ortam olustur
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Bagimliliklari kur
pip install -r requirements.txt
```

Gelistirme bagimliliklari (test, lint):

```bash
pip install -r requirements-dev.txt
# veya
make dev
```

---

## 5. Tesseract OCR (Opsiyonel)

Taranmis (goruntu bazli) PDF dosyalari icin gerekli. Text tabanli PDF'ler icin gerekmez.

### Ubuntu/Debian

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-tur
```

### Windows

[UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) adresinden Tesseract yukle. Kurulumda Turkce dil paketini sec.

---

## 6. Telegram Bot Olustur

1. Telegram'da [@BotFather](https://t.me/BotFather) ile konusma ac
2. `/newbot` komutunu gonder
3. Bot adi ve kullanici adi gir
4. BotFather'in verdigi **token**'i kaydet → `.env` dosyasina `TELEGRAM_BOT_TOKEN` olarak yaz

### Kendi Chat ID'ni Bul

1. Telegram'da [@userinfobot](https://t.me/userinfobot) ile konusma ac
2. `/start` yaz
3. Gelen **Id** degerini kaydet → `.env` dosyasina `TELEGRAM_OWNER_ID` olarak yaz

---

## 7. API Anahtarlari

Bot en az bir LLM API anahtari gerektirir. Ikisini de yapilandirabilirsin; bot task bazinda en uygun modeli secer.

### OpenAI

1. [platform.openai.com](https://platform.openai.com/api-keys) adresinde API Key olustur
2. `.env` icine `OPENAI_API_KEY=sk-...` olarak yaz

### Google Gemini

1. [aistudio.google.com/apikey](https://aistudio.google.com/apikey) adresinde API Key olustur
2. `.env` icine `GEMINI_API_KEY=...` olarak yaz

---

## 8. Bilkent Moodle Bilgileri

Bilkent her donem ayri Moodle URL'si kullanir. Guncel URL'yi Bilkent web sitesinden kontrol et.

```
# Ornek (2025-2026 Bahar)
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=22003467
MOODLE_PASSWORD=senin_sifren
```

**Not:** Bot ilk baglantida Moodle token alir ve `data/moodle_token.txt` dosyasina kaydeder. Sonraki baslatmalarda token otomatik kullanilir.

---

## 9. .env Dosyasini Doldur

```bash
cp .env.example .env
```

`.env` dosyasini ac ve su alanlari doldur:

```ini
# ZORUNLU
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=...
MOODLE_PASSWORD=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_OWNER_ID=...

# EN AZ BIRINI DOLDUR
OPENAI_API_KEY=...
GEMINI_API_KEY=...

# OPSIYONEL (varsayilanlar genelde yeterli)
# MODEL_CHAT=gemini-2.5-flash
# RAG_SIMILARITY_THRESHOLD=0.65
# LOG_LEVEL=INFO
```

Tum degiskenler ve varsayilan degerler icin `.env.example` dosyasina bak.

---

## 10. Calistir

```bash
# Sanal ortami aktif et
source venv/bin/activate

# Botu baslat
python -m bot.main
```

Veya Makefile ile:

```bash
make run
```

Basarili baslatma logu:

```
INFO | Initializing bot components...
INFO | Vector store loaded. 3661 chunks.
INFO | BM25 index built: 3661 chunks in 2.34s
INFO | Moodle connection established (courses=5)
INFO | Healthcheck endpoint listening on 0.0.0.0:8080/health
INFO | Bot started
```

---

## 11. Sunucu Kurulumu (VPS)

### Yontem A: Systemd (Onerilen)

```bash
# 1. Projeyi sunucuya kopyala
scp -r . root@SUNUCU_IP:/opt/moodle-bot/

# 2. Sunucuya baglan
ssh root@SUNUCU_IP

# 3. Bagimliliklari kur
cd /opt/moodle-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. .env dosyasini olustur
cp .env.example .env
nano .env   # Alanlari doldur

# 5. Systemd service kur
cp scripts/moodle-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable moodle-bot
systemctl start moodle-bot

# 6. Kontrol
systemctl status moodle-bot
journalctl -u moodle-bot -f
curl http://localhost:8080/health
```

### Yontem B: Docker

```bash
# 1. Projeyi sunucuya kopyala ve baglan
scp -r . root@SUNUCU_IP:/opt/moodle-bot/
ssh root@SUNUCU_IP
cd /opt/moodle-bot

# 2. .env olustur
cp .env.example .env
nano .env

# 3. Docker ile calistir
docker compose up -d
docker compose logs -f
```

### Guncellemeler

```bash
# Lokal makineden (Makefile ile)
make deploy

# Veya manuel
git push origin main
ssh root@SUNUCU_IP "cd /opt/moodle-bot && git pull && systemctl restart moodle-bot"
```

---

## 12. Sorun Giderme

### Bot baslamiyor

```bash
# Loglara bak
journalctl -u moodle-bot --no-pager -n 50
```

Yaygin sorunlar:
- `TELEGRAM_BOT_TOKEN` bos → `.env` kontrol et
- Moodle baglantisi basarisiz → `MOODLE_URL` dogru mu?
- Port 8080 kullanimda → `HEALTHCHECK_PORT` degistir

### Moodle baglantisi basarisiz

- URL donem bazli degisir. Bilkent web sitesinden guncel URL'yi kontrol et
- Token suresi dolmus olabilir → `data/moodle_token.txt` sil ve yeniden baslat

### Embedding modeli indirmiyor

```bash
# Manuel indir
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Indirdikten sonra .env'de HF_HUB_OFFLINE=1 yap
```

### Telegram Conflict hatasi

```
telegram.error.Conflict: terminated by other getUpdates request
```

Baska bir bot instance calisiyor. Oncekini durdurup bekle:

```bash
systemctl stop moodle-bot
sleep 10
systemctl start moodle-bot
```
