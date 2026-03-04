# STARS Full Cache + Exam Notifications

## Problem
1. `fetch_all_data()` fetches 6 data types but only 3 are written to SQLite cache (schedule, grades, attendance). Missing: exams, letter_grades, user_info, transcript.
2. No exam reminder notifications exist — students miss exams.
3. Exam room info comes via AIRS/DAIS mail 1-2 days before exam but is not correlated with exam data.
4. No `get_exams`, `get_transcript`, or `get_letter_grades` tools — users can't query this data.

## Design

### 1. Cache All STARS Data (6 → 7 types)

**`_populate_stars_cache()` in `bot/main.py`:**
Add to existing 3 cache writes:
- `cache_db.set_json("exams", owner_id, cache.exams)`
- `cache_db.set_json("letter_grades", owner_id, cache.letter_grades)`
- `cache_db.set_json("user_info", owner_id, cache.user_info)`

**`fetch_all_data()` in `core/stars_client.py`:**
Add transcript fetch (currently missing):
- `cache.transcript = self.get_transcript(user_id) or []`
- Then in `_populate_stars_cache`: `cache_db.set_json("transcript", owner_id, cache.transcript)`

### 2. New Background Jobs (`notification_service.py`)

#### `_sync_exams` — every 6 hours
- Fetch `stars.get_exams(OWNER_ID)` → write to `cache_db.set_json("exams", ...)`
- Same pattern as `_sync_schedule` (cache-only, no diff notification)

#### `_check_exam_reminders` — every 1 hour
- Read exams from cache: `cache_db.get_json("exams", OWNER_ID)`
- Parse each exam's date field
- If exam is tomorrow (within 24h):
  1. Search recent mails for exam room: match course code + exam keywords in mail body
  2. Extract room info via regex (Room/Salon/Sınıf + alphanumeric pattern)
  3. Send notification:
     ```
     📝 Yarın sınavın var!

     CTIS 256 — Midterm
     📅 06/03/2026, 10:00-12:00
     🏫 Salon: B-201

     Başarılar!
     ```
  4. Track sent reminders in cache to avoid duplicates: `cache_db.set_json("exam_reminders_sent", ...)`

**Mail matching logic:**
- `cache_db.get_emails(limit=30)` → search body for course code (e.g. "CTIS 256")
- If found, regex extract room: `r'(?:Room|Salon|Sınıf|Derslik)[:\s]*([A-Z]?-?\d{1,3}[A-Z]?)'`
- Fallback: send notification without room info

### 3. New Tools (`agent_service.py`)

#### `get_exams` tool
- Description: "Sınav takvimini gösterir. 'sınavlarım', 'exam schedule', 'ne zaman sınav' gibi isteklerde çağır."
- Cache fallback pattern: live fetch → cache_db → error message
- Output format: course, exam name, date, time, room (if available from mail)

#### `get_transcript` tool
- Description: "Transkript / alınan dersler ve notları gösterir."
- Cache fallback pattern

#### `get_letter_grades` tool
- Description: "Harf notlarını dönem bazlı gösterir."
- Cache fallback pattern

### 4. File Changes

| File | Change |
|------|--------|
| `core/stars_client.py` | `fetch_all_data` → add `transcript` field |
| `bot/main.py` | `_populate_stars_cache` → write all 7 types to cache |
| `bot/services/notification_service.py` | +`_sync_exams` (6h), +`_check_exam_reminders` (1h) |
| `bot/services/agent_service.py` | +`get_exams`, `get_transcript`, `get_letter_grades` tools with cache fallback |

### 5. Implementation Order
1. `stars_client.py` — add transcript to `fetch_all_data`
2. `bot/main.py` — expand `_populate_stars_cache` to 7 types
3. `notification_service.py` — add `_sync_exams` + `_check_exam_reminders`
4. `agent_service.py` — add 3 new tools with cache fallback
5. Tests — unit tests for new tools and reminder logic
6. Deploy + verify
