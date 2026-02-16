# Moodle Student Tracker

Bilkent Moodle ders materyallerini indeksleyip **Telegram uzerinden sohbet tabanli ogretim** yapan RAG (Retrieval-Augmented Generation) botu.

Ogrenci bir kurs secer, sorusunu yazar; bot ilgili ders materyallerini hybrid arama (FAISS + BM25) ile bulur ve LLM uzerinden pedagojik bir cevap uretir.

---

## Ozellikler

- **Chat-first ogretim** — soru-cevap akisiyla ders materyalinden ogrenme
- **Hybrid RAG arama** — FAISS (semantik) + BM25 (keyword) fusion via Reciprocal Rank Fusion
- **Teaching / Guidance modu** — yeterli materyal varsa ogretir, yoksa konulara yonlendirir
- **Coklu kurs destegi** — kullanici bazli aktif kurs secimi
- **Moodle senkronizasyonu** — materyalleri Moodle API'den otomatik ceker ve indeksler
- **Admin dokuman yukleme** — Telegram uzerinden PDF/DOCX/PPTX yukleyerek indeksleme
- **Konusma hafizasi** — kullanici bazli son 5 mesaj baglam olarak kullanilir
- **Rate limiting** — kullanici bazli istek sinirlamasi
- **Healthcheck** — HTTP `/health` endpointi (uptime, chunk sayisi, aktif kullanici)
- **Docker & systemd** — her iki deployment yontemi desteklenir

---

## Mimari

```
                         Telegram
                            |
                     bot/main.py
                     (Application)
                            |
              +-------------+-------------+
              |                           |
     bot/handlers/                 bot/middleware/
     commands.py                   auth.py
     messages.py                   error_handler.py
              |
     bot/services/
     rag_service.py -----> core/vector_store.py (FAISS + BM25)
     llm_service.py -----> core/llm_engine.py   (OpenAI / Gemini / GLM)
     user_service.py       core/moodle_client.py (Moodle API)
     document_service.py   core/sync_engine.py   (materyal sync)
     topic_cache.py        core/document_processor.py (PDF/DOCX/PPTX)
     conversation_memory.py
```

### Katmanlar

| Katman | Dizin | Sorumluluk |
|--------|-------|------------|
| **Handlers** | `bot/handlers/` | Telegram komut ve mesaj routing |
| **Services** | `bot/services/` | Is mantigi — RAG retrieval, LLM cagri, kullanici state |
| **Middleware** | `bot/middleware/` | Yetkilendirme, hata yakalama |
| **Config & State** | `bot/config.py`, `bot/state.py` | Typed runtime konfigurasyon, paylasimli state container |
| **Core** | `core/` | Domain logic — vektor deposu, LLM engine, Moodle client, dokuman isleme |

---

## Mesaj Akisi

```
Kullanici mesaji
    |
    v
[Rate limit kontrolu]
    |
    v
[Aktif kurs kontrolu] --x--> "Kurs secin: /courses"
    |
    v
[Hybrid RAG arama] (FAISS semantik + BM25 keyword --> RRF fusion)
    |
    v
[Yeterli materyal var mi?]
   / \
  Da  Hayir
  |     |
  v     v
Teaching   Guidance
Mode       Mode
  |         |
  v         v
LLM: materyale  LLM: mevcut konulari
dayali cevap    oner + ornek sorular
```

### Teaching Mode

Materyal yeterli oldugunda (>= `RAG_MIN_CHUNKS` chunk, similarity >= `RAG_SIMILARITY_THRESHOLD`), bot materyale dayali pedagojik cevap uretir. Hocanin terminolojisini korur, uydurma bilgi vermez.

### Guidance Mode

Materyal yetersiz oldugunda, bot teknik detay vermeden ogrenciyi mevcut konulara yonlendirir ve daha spesifik soru ornekleri sunar.

---

## Proje Yapisi

```
Moodle_Student_Tracker/
|-- bot/
|   |-- config.py               # AppConfig dataclass, .env okuma
|   |-- state.py                # BotState container (paylasimli runtime state)
|   |-- main.py                 # Uygulama giris noktasi, wiring
|   |-- logging_config.py       # Structured logging ayarlari
|   |-- exceptions.py           # Ozel exception tipleri
|   |-- handlers/
|   |   |-- commands.py         # /start, /help, /courses, /upload, /stats
|   |   +-- messages.py         # Text mesaj + dokuman upload handler
|   |-- services/
|   |   |-- rag_service.py      # Hybrid retrieval (FAISS + BM25)
|   |   |-- llm_service.py      # Teaching/guidance LLM cagrilari
|   |   |-- user_service.py     # Kullanici state, rate limit, konusma hafizasi
|   |   |-- document_service.py # Dokuman indeksleme
|   |   |-- topic_cache.py      # Kurs konu onbellegi
|   |   +-- conversation_memory.py
|   |-- middleware/
|   |   |-- auth.py             # Admin yetkilendirme
|   |   +-- error_handler.py    # Global hata yakalama
|   +-- utils/
|       |-- formatters.py       # Mesaj formatlama
|       +-- validators.py       # Girdi dogrulama
|-- core/
|   |-- vector_store.py         # FAISS + BM25 hybrid index
|   |-- llm_engine.py           # Multi-provider LLM routing
|   |-- llm_providers.py        # OpenAI/Gemini/GLM provider implementasyonlari
|   |-- moodle_client.py        # Moodle REST API client
|   |-- sync_engine.py          # Moodle -> lokal senkronizasyon
|   |-- document_processor.py   # PDF, DOCX, PPTX cikarma + chunking
|   |-- memory.py               # Konusma hafiza yoneticisi
|   |-- stars_client.py         # Bilkent STARS entegrasyonu
|   +-- webmail_client.py       # Bilkent webmail IMAP client
|-- tests/
|   |-- unit/                   # Birim testleri
|   |-- integration/            # Entegrasyon testleri
|   +-- e2e/                    # Uctan uca testler
|-- scripts/
|   |-- deploy.sh               # Lokal deploy (lint, test, push, SSH trigger)
|   |-- deploy-remote.sh        # Sunucu deploy (git pull, pip, systemd restart)
|   +-- moodle-bot.service      # Systemd unit dosyasi
|-- data/                       # Runtime veri (index, cache, indirilen dosyalar)
|-- images/                     # README ekran goruntuleri
|-- Dockerfile
|-- docker-compose.yml
|-- Makefile
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
|-- .env.example
+-- LICENSE
```

---

## Hizli Baslangic

### 1. Projeyi klonla

```bash
git clone https://github.com/onurcangnc/Moodle_Student_Tracker.git
cd Moodle_Student_Tracker
```

### 2. Sanal ortam olustur

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Bagimliliklari kur

```bash
pip install -r requirements.txt
```

### 4. .env dosyasini yapilandir

```bash
cp .env.example .env
# .env icindeki alanlari doldur (detaylar icin SETUP.md)
```

Minimum gerekli alanlar:
- `MOODLE_URL`, `MOODLE_USERNAME`, `MOODLE_PASSWORD`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_OWNER_ID`
- En az bir LLM API anahtari (`OPENAI_API_KEY` veya `GEMINI_API_KEY`)

### 5. Calistir

```bash
python -m bot.main
# veya
make run
```

---

## Komutlar

| Komut | Aciklama | Yetki |
|-------|----------|-------|
| `/start` | Botu baslatir, karsilama mesaji | Herkes |
| `/help` | Kullanim rehberi | Herkes |
| `/courses` | Kurslari listeler, aktif kurs secer | Herkes |
| `/courses <ad>` | Belirtilen kursu aktif kurs olarak secer | Herkes |
| `/upload` | Dokuman yukleme modu acar (sonraki dosya indekslenir) | Admin |
| `/stats` | Bot istatistikleri (chunk, kurs, dosya sayisi) | Admin |

**Normal kullanim:** `/courses` ile kurs sec, sonra sorunu mesaj olarak yaz.

---

## Konfigurasyon

Tum konfigurasyon `.env` dosyasindan okunur. Ornek icin `.env.example` dosyasina bakin.

### Temel degiskenler

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `MOODLE_URL` | — | Bilkent Moodle URL (donem bazli degisir) |
| `MOODLE_USERNAME` | — | Moodle kullanici adi |
| `MOODLE_PASSWORD` | — | Moodle sifresi |
| `TELEGRAM_BOT_TOKEN` | — | @BotFather'dan alinan token |
| `TELEGRAM_OWNER_ID` | — | Bot sahibinin Telegram chat ID'si |
| `TELEGRAM_ADMIN_IDS` | — | Ek admin ID'leri (virgul ayirmali) |

### LLM Routing

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `OPENAI_API_KEY` | — | OpenAI API anahtari |
| `GEMINI_API_KEY` | — | Google Gemini API anahtari |
| `MODEL_CHAT` | `gemini-2.5-flash` | Ana sohbet modeli |
| `MODEL_STUDY` | `gemini-2.5-flash` | Study mode modeli |
| `MODEL_EXTRACTION` | `gpt-4.1-nano` | Hafiza cikarma |
| `MODEL_SUMMARY` | `gemini-2.5-flash` | Ozet uretimi |

### RAG Parametreleri

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `RAG_SIMILARITY_THRESHOLD` | `0.65` | Minimum benzerlik skoru |
| `RAG_MIN_CHUNKS` | `2` | Teaching mode icin minimum chunk |
| `RAG_TOP_K` | `5` | Her aramada dondurulecek chunk sayisi |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding modeli |
| `CHUNK_SIZE` | `1000` | Chunk boyutu (karakter) |
| `CHUNK_OVERLAP` | `200` | Chunk overlapi |

### Operasyonel

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `RATE_LIMIT_MAX` | `30` | Pencere basina max istek |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit penceresi (saniye) |
| `MEMORY_MAX_MESSAGES` | `5` | Konusma hafizasindaki max mesaj |
| `MEMORY_TTL_MINUTES` | `30` | Hafiza TTL (dakika) |
| `HEALTHCHECK_ENABLED` | `true` | HTTP health endpoint |
| `HEALTHCHECK_PORT` | `8080` | Health endpoint portu |
| `LOG_LEVEL` | `INFO` | Log seviyesi |

---

## Deployment

### Systemd (onerilen)

```bash
# Service dosyasini kopyala
sudo cp scripts/moodle-bot.service /etc/systemd/system/moodle-bot.service

# Baslatma
sudo systemctl daemon-reload
sudo systemctl enable moodle-bot
sudo systemctl start moodle-bot

# Log takibi
journalctl -u moodle-bot -f
```

### Docker

```bash
# .env dosyasini ayarla
cp .env.example .env

# Build ve calistir
docker compose up -d

# Log takibi
docker compose logs -f
```

### Health Check

```bash
curl http://localhost:8080/health
# {"status": "ok", "uptime_seconds": 3600, "version": "abc1234", "chunks_loaded": 3661, "active_users_24h": 5}
```

---

## Gelistirme

```bash
# Gelistirme bagimliklarini kur
make dev

# Lint
make lint

# Format
make format

# Unit testleri calistir
make test

# Tum testleri calistir
make test-all

# Coverage raporu
make test-cov

# Cache temizle
make clean

# Sunucuya deploy
make deploy
```

---

## Tech Stack

| Bilesen | Teknoloji |
|---------|-----------|
| Runtime | Python >= 3.10 |
| Telegram | python-telegram-bot 21.x (asyncio) |
| LLM | OpenAI API, Google Gemini, GLM (task-based routing) |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Vector Search | FAISS (CPU) |
| Keyword Search | rank-bm25 (Snowball TR/EN stemming) |
| Search Fusion | Reciprocal Rank Fusion (k=60) |
| PDF Extraction | PyMuPDF + pymupdf4llm + Tesseract OCR |
| DOCX/PPTX | python-docx, python-pptx |
| Chunking | langchain-text-splitters (RecursiveCharacterTextSplitter) |
| Config | python-dotenv |
| Lint/Format | ruff |
| Test | pytest, pytest-asyncio, pytest-cov |

---

## Ekran Goruntuleri

| | |
|---|---|
| ![Kurs secimi](images/1.png) | ![Soru-cevap](images/2.png) |
| ![Materyal listesi](images/3.png) | ![Teaching mode](images/4.png) |
| ![Moodle entegrasyonu](images/moodle1.png) | ![Senkronizasyon](images/moodle2.png) |

---

## Lisans

MIT License. Detaylar icin [LICENSE](LICENSE) dosyasina bakin.
