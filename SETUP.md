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
12. [Guncelleme ve Bakim](#12-guncelleme-ve-bakim)
13. [Sorun Giderme](#13-sorun-giderme)

---

## 1. Gereksinimler

| Gereksinim | Minimum | Tavsiye |
|-----------|---------|---------|
| Python | 3.10 | 3.12 |
| RAM | 2 GB | 4 GB |
| Disk | 2 GB | 5 GB |
| OS | Ubuntu 22.04 / Windows 10 | Ubuntu 24.04 |

**Notlar:**
- Embedding modeli (`paraphrase-multilingual-MiniLM-L12-v2`) ilk calistirmada ~500 MB indirir.
- OCR kullanacaksan Tesseract icin ek ~100 MB disk alani gerekir.
- Sunucu deploy icin en az 2 GB RAM gerekli (embedding modeli bellekte kalir).

---

## 2. Python Kurulumu

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
python3 --version   # 3.10+
```

### Windows

1. [python.org](https://www.python.org/downloads/) adresinden Python 3.10+ indirip kur
2. Kurulum sirasinda **"Add Python to PATH"** secenegini isaretlemeyi unutma
3. Dogrulama:

```powershell
python --version   # 3.10+
```

### macOS

```bash
brew install python@3.12
python3 --version   # 3.10+
```

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

# Sanal ortami aktif et
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows (PowerShell)
# venv\Scripts\activate.bat     # Windows (CMD)

# Bagimliliklari kur
pip install -r requirements.txt
```

Gelistirme bagimlilikari (test, lint):

```bash
pip install -r requirements-dev.txt
# veya
make dev
```

### PyStemmer (Performans)

BM25 arama indeksi Snowball stemmer kullanir. `snowballstemmer` paketi pure Python fallback ile calisir ama yavas olabilir (~30s). C uzantisi PyStemmer ile ~1.5s'ye duser:

```bash
pip install PyStemmer
```

> **Not:** PyStemmer `requirements.txt` icinde degildir (C derleyici gerektirir). Kurulamazsa bot pure Python ile calismaya devam eder, sadece baslatma suresi uzar.

---

## 5. Tesseract OCR (Opsiyonel)

Taranmis (goruntu bazli) PDF dosyalari icin gerekli. Text tabanli PDF'ler icin gerekmez. Bot, her PDF'i sayfa sayfa analiz eder ve sadece taranmis sayfalari OCR'a gonderir.

### Ubuntu/Debian

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-tur
tesseract --version
```

### Windows

1. [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) adresinden Tesseract yukle
2. Kurulumda Turkce dil paketini sec
3. Tesseract'i PATH'e ekle veya `.env` icinde `TESSERACT_CMD` belirt

### macOS

```bash
brew install tesseract tesseract-lang
```

---

## 6. Telegram Bot Olustur

### 6.1. Bot Token Al

1. Telegram'da [@BotFather](https://t.me/BotFather) ile konusma ac
2. `/newbot` komutunu gonder
3. Bot adi gir (ornek: `Moodle Asistan`)
4. Kullanici adi gir (ornek: `moodle_asistan_bot`)
5. BotFather'in verdigi **token**'i kaydet

   ```
   5123456789:ABCdefGHIjklMNOpqrs-TUVwxyz12345
   ```

### 6.2. Kendi Chat ID'ni Bul

1. Telegram'da [@userinfobot](https://t.me/userinfobot) ile konusma ac
2. `/start` yaz
3. Gelen **Id** degerini kaydet (ornek: `123456789`)

Bu ID'yi `.env` dosyasinda `TELEGRAM_OWNER_ID` olarak kullanacaksin. Owner ID'ye sahip kullanici admin komutlarini (/upload, /stats) kullanabilir.

### 6.3. Ek Admin Ekleme (Opsiyonel)

Birden fazla kisi admin komutlarini kullanacaksa, her birinin chat ID'sini virgul ayirarak `.env` icinde belirt:

```
TELEGRAM_ADMIN_IDS=111222333,444555666
```

---

## 7. API Anahtarlari

Bot en az bir LLM API anahtari gerektirir. Birden fazla provider yapilandirilabilir; bot gorev bazinda en uygun modeli secer (`TaskRouter`).

### 7.1. Google Gemini (Onerilen)

Ana sohbet modeli olarak Gemini 2.5 Flash kullanilir. Ucretsiz katmanda sinirli (5 RPM / 20 RPD).

1. [aistudio.google.com/apikey](https://aistudio.google.com/apikey) adresine git
2. "Create API Key" tikla
3. `.env` icine ekle:

   ```
   GEMINI_API_KEY=AIzaSy...
   ```

### 7.2. OpenAI

Hafiza cikarma ve konu tespiti icin GPT-4.1 nano kullanilir.

1. [platform.openai.com/api-keys](https://platform.openai.com/api-keys) adresine git
2. "Create new secret key" tikla
3. `.env` icine ekle:

   ```
   OPENAI_API_KEY=sk-proj-...
   ```

### 7.3. Model Routing (Opsiyonel)

Her gorev icin hangi modelin kullanilacagini `.env` ile degistirebilirsin:

```ini
MODEL_CHAT=gemini-2.5-flash        # Ana sohbet (RAG)
MODEL_STUDY=gemini-2.5-flash       # Derin ogretim modu
MODEL_EXTRACTION=gpt-4.1-nano      # Hafiza cikarma
MODEL_TOPIC_DETECT=gpt-4.1-nano    # Konu tespiti
MODEL_SUMMARY=gemini-2.5-flash     # Haftalik ozet
MODEL_QUESTIONS=gemini-2.5-flash   # Pratik sorular
MODEL_OVERVIEW=gemini-2.5-flash    # Kurs genel bakis
```

Desteklenen modeller: Gemini 2.5 Flash/Pro, GPT-4.1 nano/mini, GPT-5 mini, GLM 4.5/4.7, Claude Haiku/Sonnet/Opus.

---

## 8. Bilkent Moodle Bilgileri

Bilkent her donem ayri Moodle URL'si kullanir. Guncel URL'yi Bilkent web sitesinden kontrol et.

```
# Ornek (2025-2026 Bahar)
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=22003467
MOODLE_PASSWORD=senin_sifren
```

**Onemli notlar:**
- Bot ilk baglantida Moodle token alir ve `data/moodle_token.txt` dosyasina kaydeder. Sonraki baslatmalarda token otomatik kullanilir.
- Donem degistiginde `MOODLE_URL`'yi guncellemen gerekir.
- Moodle sifreni degistirirsen `data/moodle_token.txt` dosyasini sil ve botu yeniden baslat.

---

## 9. .env Dosyasini Doldur

```bash
cp .env.example .env
```

`.env` dosyasini ac ve su alanlari doldur:

```ini
# ── ZORUNLU ─────────────────────────────────────────
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=...
MOODLE_PASSWORD=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_OWNER_ID=...

# ── EN AZ BIRINI DOLDUR ────────────────────────────
OPENAI_API_KEY=...
GEMINI_API_KEY=...

# ── OPSIYONEL (varsayilanlar genelde yeterli) ──────
# MODEL_CHAT=gemini-2.5-flash
# RAG_SIMILARITY_THRESHOLD=0.65
# HEALTHCHECK_PORT=9090
# LOG_LEVEL=INFO
```

Tum degiskenler ve varsayilan degerler icin [.env.example](.env.example) dosyasina bak.

### Konfigürasyon Referansi

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `RATE_LIMIT_MAX` | `30` | Pencere basina max istek |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit penceresi (saniye) |
| `MEMORY_MAX_MESSAGES` | `5` | Konusma hafizasindaki max mesaj |
| `MEMORY_TTL_MINUTES` | `30` | Hafiza TTL (dakika) |
| `HEALTHCHECK_PORT` | `9090` | Health endpoint portu |
| `HEALTHCHECK_ENABLED` | `true` | Health endpoint acik/kapali |
| `CHUNK_SIZE` | `1000` | Metin chunk boyutu (karakter) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (karakter) |
| `RAG_TOP_K` | `5` | Aramada dondurulecek max chunk |

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

Basarili baslatma ciktisi:

```
INFO | Initializing bot components...
INFO | Vector store loaded. 3661 chunks.
INFO | BM25 index built: 3661 chunks in 1.62s
INFO | Moodle connection established (courses=5)
INFO | Healthcheck endpoint listening on 0.0.0.0:9090/health
INFO | Bot started
```

### Ilk Kullanim

1. Telegram'da botunuzla konusma acin
2. `/start` yazip karsilama mesajini gorun
3. `/courses` ile yuklu kurslari listeleyin
4. Bir kurs secin: `/courses CTIS 363`
5. Sorunuzu yazin: `Etik nedir?`
6. Bot materyallerden cevap uretecek

### Health Check

Bot bir HTTP health endpoint sunar:

```bash
curl http://localhost:9090/health
```

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "version": "abc1234",
  "chunks_loaded": 3661,
  "active_users_24h": 1
}
```

---

## 11. Sunucu Kurulumu (VPS)

### Yontem A: Systemd (Onerilen)

```bash
# 1. Projeyi sunucuya kopyala
scp -r . root@SUNUCU_IP:/opt/moodle-bot/

# 2. Sunucuya baglan
ssh root@SUNUCU_IP

# 3. Bot kullanicisi olustur (guvenlik icin)
useradd -r -s /bin/false botuser
chown -R botuser:botuser /opt/moodle-bot

# 4. Bagimliliklari kur
cd /opt/moodle-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install PyStemmer   # BM25 performansi icin onerilen

# 5. .env dosyasini olustur
cp .env.example .env
nano .env   # Alanlari doldur

# 6. Systemd service kur
cp scripts/moodle-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable moodle-bot
systemctl start moodle-bot

# 7. Kontrol
systemctl status moodle-bot
journalctl -u moodle-bot -f
curl http://localhost:9090/health
```

#### Systemd Service Detaylari

`scripts/moodle-bot.service` dosyasi guvenlik hardening icin su korumalar icerir:

| Direktif | Aciklama |
|----------|----------|
| `NoNewPrivileges=true` | Yeni ayricalik kazanimini engeller |
| `ProtectSystem=strict` | Sistem dizinlerini salt okunur yapar |
| `ProtectHome=true` | /home erisimini engeller |
| `ReadWritePaths=...` | Sadece data/ ve logs/ yazilabilir |
| `PrivateTmp=true` | Izole gecici dizin kullanir |

> **Not:** Service dosyasi `User=botuser` kullanir. Root ile calismak istiyorsaniz bu satirlari yorum satirina alin veya kendi kullanici adinizi yazin.

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

# 4. Kontrol
docker compose logs -f
docker compose ps
```

#### Docker Detaylari

| Dosya | Icerik |
|-------|--------|
| `Dockerfile` | Python 3.11-slim bazli, `python -m bot.main` entry point |
| `docker-compose.yml` | Health check (30s interval), persistent volume (`bot_data`) |

Docker volume `bot_data`, FAISS indeksi, indirilen dosyalar ve konusma gecmisini saklar. Container yeniden olusturulsa bile veriler korunur.

---

## 12. Guncelleme ve Bakim

### Makefile ile Deploy

```bash
# Tam deploy: lint → test → push → SSH deploy
make deploy
```

Bu komut sirasiyla:
1. `ruff check .` — lint kontrolu
2. `pytest tests/unit/` — birim testleri
3. `git push origin main` — uzak repo'ya push
4. SSH ile sunucuda `scripts/deploy-remote.sh` calistirir

### Manuel Guncelleme

```bash
# Lokal degisiklikleri push et
git push origin main

# Sunucuda guncelle
ssh root@SUNUCU_IP "cd /opt/moodle-bot && git pull && systemctl restart moodle-bot"
```

### Makefile Komutlari

| Komut | Aciklama |
|-------|----------|
| `make install` | Production bagimlilikari kur |
| `make dev` | Gelistirme bagimlilikari kur |
| `make run` | Botu baslat |
| `make test` | Unit testleri calistir |
| `make test-all` | Tum testleri calistir |
| `make test-cov` | Coverage raporu |
| `make lint` | Ruff lint kontrolu |
| `make format` | Ruff otomatik format |
| `make deploy` | Sunucuya deploy |
| `make logs` | Sunucu loglarini goster |
| `make status` | Sunucu servis durumu |
| `make restart` | Sunucu servisi yeniden baslat |
| `make health` | Sunucu health check |
| `make clean` | Cache dosyalarini temizle |

### Indeksi Sifirdan Olusturma

Eger RAG indeksinde sorun yasarsan:

```bash
# Sunucuda
cd /opt/moodle-bot/data
rm -f faiss.index metadata.json sync_state.json
systemctl restart moodle-bot
# Bot baslatildiginda Moodle'dan materyalleri tekrar ceker ve indeksler
```

### Embedding Modeli Offline Modu

Ilk calistirmada model Hugging Face'den indirilir (~500 MB). Indirdikten sonra offline moda gecebilirsin:

```ini
# .env
HF_HUB_OFFLINE=1
```

---

## 13. Sorun Giderme

### Bot baslamiyor

```bash
# Loglara bak
journalctl -u moodle-bot --no-pager -n 50
# veya Docker ile:
docker compose logs --tail 50
```

Yaygin sorunlar:

| Hata | Cozum |
|------|-------|
| `TELEGRAM_BOT_TOKEN bos` | `.env` dosyasinda token'i kontrol et |
| `Moodle connection failed` | `MOODLE_URL` dogru mu? Donem bazli degisir |
| `Port 9090 already in use` | `HEALTHCHECK_PORT` degistir veya `HEALTHCHECK_ENABLED=false` yap |
| `No module named 'bot'` | Projenin kok dizininde misin? `python -m bot.main` kullan |
| `ModuleNotFoundError` | Sanal ortam aktif mi? `source venv/bin/activate` |

### LLM API Hatasi

```bash
# OpenAI baglantisini test et
python -c "from openai import OpenAI; c = OpenAI(); print(c.models.list().data[0].id)"

# Gemini baglantisini test et
python -c "
from openai import OpenAI
c = OpenAI(api_key='GEMINI_KEY', base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
print(c.models.list().data[0].id)
"
```

| Hata | Cozum |
|------|-------|
| `AuthenticationError` | API anahtari yanlis veya suresi dolmus |
| `RateLimitError` | Gemini free tier: 5 RPM, 20 RPD. Ucretli plana gec veya OpenAI kullan |
| `proxies TypeError` | `openai` paketini guncelle: `pip install 'openai>=1.58.0'` |

### Moodle Baglantisi Basarisiz

- URL donem bazli degisir. Bilkent web sitesinden guncel URL'yi kontrol et
- Token suresi dolmus olabilir:

```bash
rm data/moodle_token.txt
# Botu yeniden baslat — yeni token otomatik alinir
```

### Embedding Modeli Indirmiyor

```bash
# Manuel indir
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Indirdikten sonra .env'de offline moda gec
# HF_HUB_OFFLINE=1
```

### Telegram Conflict Hatasi

```
telegram.error.Conflict: terminated by other getUpdates request
```

Baska bir bot instance calisiyor veya onceki polling oturumu henuz kapanmamis:

```bash
# Systemd
systemctl stop moodle-bot
sleep 15
systemctl start moodle-bot

# Docker
docker compose down
sleep 15
docker compose up -d
```

### BM25 Indeksi Yavas

BM25 indeks olusturma 20+ saniye suruyorsa:

```bash
# PyStemmer C uzantisi kur
pip install PyStemmer
# Botu yeniden baslat — ~1.5s'ye dusecek
```

### Health Check Calismiyorsa

```bash
# Port kontrolu
ss -tlnp | grep 9090
# veya
netstat -tlnp | grep 9090

# Manuel test
curl -v http://localhost:9090/health

# Devre disi birak (istege bagli)
# .env icinde: HEALTHCHECK_ENABLED=false
```
