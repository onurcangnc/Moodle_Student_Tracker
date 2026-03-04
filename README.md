# Moodle Student Tracker

![Bilkent + Moodle](images/1.png)

Bilkent Moodle ders materyallerini indeksleyip **Telegram uzerinden sohbet tabanli ogretim** yapan RAG (Retrieval-Augmented Generation) botu.

Ogrenci bir kurs secer, sorusunu yazar; bot ilgili ders materyallerini **hybrid arama** (FAISS + BM25) ile bulur ve LLM uzerinden pedagojik bir cevap uretir. Materyalde olmayan bilgiyi uydurma yerine ogrenciyi dogru konulara yonlendirir.

---

## Ozellikler

### Aktif

- **Chat-first ogretim** — kurs sec, soru sor, materyalden ogren
- **Hybrid RAG** — FAISS (semantik) + BM25 (keyword), Reciprocal Rank Fusion ile birlestirme
- **Teaching / Guidance modu** — yeterli materyal varsa ogretir, yoksa konulara yonlendirir
- **Coklu kurs destegi** — kullanici bazli aktif kurs secimi
- **Moodle senkronizasyonu** — materyalleri Moodle API'den otomatik ceker ve indeksler
- **Admin dokuman yukleme** — Telegram uzerinden PDF/DOCX/PPTX yukleyerek indeksleme
- **Multi-provider LLM** — Gemini, OpenAI, GLM arasinda task bazli model routing
- **Konusma hafizasi** — kullanici bazli son mesajlar baglam olarak kullanilir
- **Rate limiting** — kullanici bazli istek sinirlamasi
- **Healthcheck** — HTTP `/health` endpointi (uptime, chunk sayisi, aktif kullanici)
- **Docker & systemd** — her iki deployment yontemi desteklenir

### Planlanan (core modulleri hazir, handler baglantisi yapilacak)

- **STARS entegrasyonu** — notlar, devamsizlik, sinav takvimi sorgulama
- **Webmail ozeti** — Bilkent mail kutusundan hoca maillerini ozetleme
- **Odev takibi** — yaklasan deadline bildirimleri ve manuel sorgulama
- **Arka plan bildirimleri** — STARS degisiklik, yeni odev, yeni mail notification

---

## Mimari

### Genel Yapi

Bot **katmanli mimari** (layered architecture) kullanir. Her katman sadece altindaki katmanla iletisim kurar:

```
                          Telegram API
                              |
                       bot/main.py
                       (Application wiring)
                              |
               +--------------+--------------+
               |                             |
      bot/handlers/                   bot/middleware/
      commands.py                     auth.py
      messages.py                     error_handler.py
               |
      bot/services/
      rag_service.py ---------> core/vector_store.py  (FAISS + BM25)
      llm_service.py ---------> core/llm_engine.py    (Multi-provider LLM)
      user_service.py            core/llm_providers.py  (Adapter pattern)
      document_service.py        core/moodle_client.py  (Moodle REST API)
      topic_cache.py             core/sync_engine.py    (Materyal pipeline)
      conversation_memory.py     core/document_processor.py (PDF/DOCX/PPTX)
```

### Katmanlar

| Katman | Dizin | Sorumluluk |
|--------|-------|------------|
| **Handlers** | `bot/handlers/` | Telegram komut ve mesaj routing |
| **Services** | `bot/services/` | Is mantigi — RAG retrieval, LLM cagri, kullanici state |
| **Middleware** | `bot/middleware/` | Yetkilendirme (admin gate), global hata yakalama |
| **Config** | `bot/config.py` | Typed `AppConfig` dataclass, tum `.env` degiskenleri |
| **State** | `bot/state.py` | Paylasimli `BotState` container (singleton) |
| **Core** | `core/` | Domain logic — vektor deposu, LLM engine, Moodle client |

### Design Patterns

| Pattern | Kullanim | Dosya |
|---------|----------|-------|
| **Adapter** | Her LLM provider icin ortak interface (`LLMAdapter` ABC) | `core/llm_providers.py` |
| **Strategy** | Task-based model routing (`TaskRouter`) | `core/llm_providers.py` |
| **Facade** | `LLMEngine` — RAG + memory + prompt yonetimini tek interface altinda toplar | `core/llm_engine.py` |
| **Singleton** | `CONFIG`, `STATE` — uygulama genelinde tek instance | `bot/config.py`, `bot/state.py` |
| **Service Layer** | Handler → Service → Core katman ayirimi | `bot/services/` |
| **Chain of Responsibility** | Fallback model zinciri (primary → fallback chain) | `core/llm_providers.py` |
| **Observer** | `post_init` hook ile Telegram command menu guncelleme | `bot/handlers/commands.py` |

---

## Mesaj Akisi

```
Kullanici mesaji
    |
    v
[Rate limit kontrolu] ---x---> "Cok hizli mesaj gonderdiniz"
    |
    v
[Aktif kurs kontrolu] ---x---> "Kurs secin: /courses"
    |
    v
[Konusma gecmisi yukle]
    |
    v
[Hybrid RAG arama]
  FAISS (semantik) + BM25 (keyword)
  --> Reciprocal Rank Fusion (k=60)
  --> Adaptive threshold: max(top_score * 0.60, 0.20)
    |
    v
[Yeterli materyal var mi?]
  (chunk sayisi >= RAG_MIN_CHUNKS
   ve similarity >= RAG_SIMILARITY_THRESHOLD)
   /              \
  Evet            Hayir
  |                 |
  v                 v
Teaching          Guidance
Mode              Mode
  |                 |
  v                 v
LLM: materyale    LLM: mevcut konulari
dayali pedagojik  oner + ornek sorular
cevap uretir      ile yonlendirir
  |                 |
  +--------+--------+
           |
           v
  [Konusma gecmisine kaydet]
           |
           v
  [Markdown ile Telegram'a gonder]
```

### Teaching Mode

Materyal yeterli oldugunda bot, hocanin terminolojisini koruyarak materyale dayali pedagojik cevap uretir. Kaynak dosya adlari `[dosya.pdf]` etiketiyle belirtilir. Materyalde olmayan bilgiyi uydurma yerine "bu konu materyalde yer almiyor" der.

![Teaching mode ornegi](images/5.png)

### Guidance Mode

Materyal yetersiz oldugunda bot, teknik detay vermeden ogrenciyi mevcut konulara yonlendirir ve daha spesifik soru ornekleri sunar.

---

## Ekran Goruntuleri

| Ogretim Modu | Materyal Secimi |
|:---:|:---:|
| ![Adim adim ogretim](images/moodle1.png) | ![Kaynak secimi](images/moodle3.png) |

| RAG ile Ders Anlatimi | Yaklasan Sinavlar |
|:---:|:---:|
| ![RAG cevap](images/moodle2.png) | ![Sinav takvimi](images/3.png) |

| Devamsizlik Bilgisi | Notlar |
|:---:|:---:|
| ![Devamsizlik](images/4.png) | ![Not durumu](images/6.jpeg) |

| Mail Ozetleri | Edebiyat Dersi Ogretimi |
|:---:|:---:|
| ![Mail ozet](images/2.jpeg) | ![Edebiyat](images/5.png) |

> **Not:** Devamsizlik, notlar ve mail ozetleri ekran goruntuleri botun onceki surumundendir. Bu ozellikler `core/` katmaninda mevcuttur ve gelecek surumde yeniden aktif edilecektir.

---

## Proje Yapisi

```
Moodle_Student_Tracker/
|-- bot/                           # Telegram bot runtime
|   |-- main.py                    # Uygulama giris noktasi, component wiring
|   |-- config.py                  # AppConfig dataclass (.env okuma)
|   |-- state.py                   # BotState container (paylasimli runtime state)
|   |-- logging_config.py          # Structured logging
|   |-- exceptions.py              # Ozel exception tipleri
|   |-- handlers/
|   |   |-- commands.py            # /start, /help, /courses, /upload, /stats
|   |   +-- messages.py            # Text mesaj → RAG akisi + dokuman upload
|   |-- services/
|   |   |-- rag_service.py         # Hybrid retrieval (FAISS + BM25 → RRF)
|   |   |-- llm_service.py         # Teaching / Guidance LLM cagrilari
|   |   |-- user_service.py        # Kurs secimi, rate limit, konusma hafizasi
|   |   |-- document_service.py    # Dokuman indeksleme
|   |   |-- topic_cache.py         # Kurs konu onbellegi
|   |   +-- conversation_memory.py # Kisa sureli konusma hafizasi
|   |-- middleware/
|   |   |-- auth.py                # Admin yetkilendirme (owner_id + admin_ids)
|   |   +-- error_handler.py       # Global exception handler
|   +-- utils/
|       |-- formatters.py          # Mesaj formatlama
|       +-- validators.py          # Girdi dogrulama
|-- core/                          # Domain logic (Telegram'dan bagimsiz)
|   |-- vector_store.py            # FAISS + BM25 hybrid index
|   |-- llm_engine.py              # Multi-provider LLM + RAG + memory
|   |-- llm_providers.py           # Provider adapter'lari (OpenAI, Gemini, GLM)
|   |-- moodle_client.py           # Moodle REST API client
|   |-- sync_engine.py             # Moodle → lokal senkronizasyon pipeline
|   |-- document_processor.py      # PDF, DOCX, PPTX cikarma + chunking
|   |-- memory.py                  # Persistent konusma hafiza yoneticisi
|   |-- stars_client.py            # Bilkent STARS entegrasyonu (OAuth + SMS 2FA)
|   |-- webmail_client.py          # Bilkent webmail IMAP client
|   +-- config.py                  # Core konfigurasyon
|-- tests/
|   |-- unit/                      # 76 birim testi
|   |-- integration/               # Entegrasyon testleri
|   +-- e2e/                       # Uctan uca testler
|-- scripts/
|   |-- deploy.sh                  # Lokal deploy (lint, test, push, SSH trigger)
|   |-- deploy-remote.sh           # Sunucu deploy (git pull, pip, restart)
|   +-- moodle-bot.service         # Systemd unit dosyasi (hardened)
|-- data/                          # Runtime veri (index, cache, indirilen dosyalar)
|-- images/                        # Dokumantasyon ekran goruntuleri
|-- Dockerfile                     # Python 3.11-slim container
|-- docker-compose.yml             # Health check + persistent volume
|-- Makefile                       # install, test, lint, deploy, health
|-- pyproject.toml                 # Ruff + pytest konfigurasyon
|-- requirements.txt               # Production bagimliliklari
|-- requirements-dev.txt           # Gelistirme bagimliliklari (pytest, ruff)
|-- .env.example                   # Ornek konfigurasyon sablonu
+-- LICENSE                        # MIT
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
# veya
make install
```

### 4. .env dosyasini yapilandir

```bash
cp .env.example .env
```

Minimum gerekli alanlar:

| Degisken | Aciklama |
|----------|----------|
| `MOODLE_URL` | Bilkent Moodle URL (donem bazli) |
| `MOODLE_USERNAME` | Moodle kullanici adi |
| `MOODLE_PASSWORD` | Moodle sifresi |
| `TELEGRAM_BOT_TOKEN` | @BotFather'dan alinan token |
| `TELEGRAM_OWNER_ID` | Bot sahibinin Telegram chat ID'si |
| `OPENAI_API_KEY` veya `GEMINI_API_KEY` | En az bir LLM API anahtari |

Detayli kurulum icin [SETUP.md](SETUP.md) dosyasina bakin.

### 5. Calistir

```bash
python -m bot.main
# veya
make run
```

Basarili cikti:

```
INFO | Initializing bot components...
INFO | Vector store loaded. 3661 chunks.
INFO | BM25 index built: 3661 chunks in 1.22s
INFO | Moodle connection established (courses=5)
INFO | Healthcheck endpoint listening on 0.0.0.0:9090/health
INFO | Bot started
```

---

## Komutlar

| Komut | Aciklama | Yetki |
|-------|----------|-------|
| `/start` | Karsilama mesaji ve kullanim rehberi | Herkes |
| `/help` | Adim adim kullanim kilavuzu | Herkes |
| `/courses` | Yuklu kurslari listeler | Herkes |
| `/courses <ad>` | Belirtilen kursu aktif kurs olarak secer | Herkes |
| `/upload` | Dokuman yukleme modunu acar (sonraki dosya indekslenir) | Admin |
| `/stats` | Bot istatistikleri (chunk, kurs, dosya sayisi) | Admin |

**Kullanim akisi:** `/courses` → kurs sec → mesaj yaz → materyale dayali cevap al.

---

## Konfigurasyon

Tum konfigurasyon `.env` dosyasindan okunur. Tam sablon: [.env.example](.env.example)

### Temel

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `MOODLE_URL` | — | Bilkent Moodle URL (donem bazli degisir) |
| `MOODLE_USERNAME` | — | Moodle kullanici adi |
| `MOODLE_PASSWORD` | — | Moodle sifresi |
| `TELEGRAM_BOT_TOKEN` | — | @BotFather'dan alinan token |
| `TELEGRAM_OWNER_ID` | — | Bot sahibinin Telegram chat ID |
| `TELEGRAM_ADMIN_IDS` | — | Ek admin ID'leri (virgul ayirmali) |

### LLM Model Routing

Bot her gorevi farkli modele yonlendirebilir. Varsayilan routing:

| Degisken | Varsayilan | Gorev |
|----------|------------|-------|
| `MODEL_CHAT` | `gemini-2.5-flash` | Ana sohbet (RAG) |
| `MODEL_STUDY` | `gemini-2.5-flash` | Derin ogretim modu |
| `MODEL_EXTRACTION` | `gpt-4.1-nano` | Hafiza cikarma |
| `MODEL_SUMMARY` | `gemini-2.5-flash` | Haftalik ozet |

Desteklenen modeller: Gemini 2.5 Flash/Pro, GPT-4.1 nano/mini, GPT-5 mini, GLM 4.5/4.7, Claude Haiku/Sonnet/Opus.

### RAG Parametreleri

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `RAG_SIMILARITY_THRESHOLD` | `0.65` | Minimum benzerlik skoru |
| `RAG_MIN_CHUNKS` | `2` | Teaching mode icin minimum chunk |
| `RAG_TOP_K` | `5` | Her aramada dondurulecek chunk sayisi |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding modeli (384 dim, 50+ dil) |
| `CHUNK_SIZE` | `1000` | Chunk boyutu (karakter) |
| `CHUNK_OVERLAP` | `200` | Chunk overlapi |

### Operasyonel

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| `RATE_LIMIT_MAX` | `30` | Pencere basina max istek |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit penceresi (saniye) |
| `MEMORY_MAX_MESSAGES` | `5` | Konusma hafizasindaki max mesaj |
| `MEMORY_TTL_MINUTES` | `30` | Hafiza TTL (dakika) |
| `HEALTHCHECK_PORT` | `9090` | Health endpoint portu |
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
# .env icerigini doldur

# Build ve calistir
docker compose up -d

# Log takibi
docker compose logs -f
```

### Health Check

```bash
curl http://localhost:9090/health
```

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "version": "abc1234",
  "chunks_loaded": 3661,
  "active_users_24h": 5
}
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

# Unit testleri calistir (76 test)
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
| LLM | OpenAI, Google Gemini, GLM, Anthropic (task-based routing) |
| Embedding | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim) |
| Semantic Search | FAISS-CPU |
| Keyword Search | rank-bm25 (Snowball TR/EN stemming via PyStemmer) |
| Search Fusion | Reciprocal Rank Fusion (k=60) |
| PDF Extraction | PyMuPDF + pymupdf4llm + Tesseract OCR (hybrid: text + scanned pages) |
| DOCX/PPTX | python-docx, python-pptx |
| Chunking | langchain-text-splitters (`RecursiveCharacterTextSplitter`) |
| Config | python-dotenv, typed dataclass |
| Lint/Format | ruff |
| Test | pytest, pytest-asyncio, pytest-cov (76 unit tests) |
| Deploy | Docker, systemd, Makefile |

---

## Lisans

MIT License. Detaylar icin [LICENSE](LICENSE) dosyasina bakin.
