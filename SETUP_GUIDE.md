# Moodle AI Assistant — Setup Guide

Follow the steps below to get everything up and running from scratch.

---

## STEP 1: Set Up the Project

```bash
# 1. Unzip and enter the directory
unzip moodle-ai-assistant.zip
cd moodle-ai-assistant

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> Warning: `sentence-transformers` will download a ~500MB model on first run (all-MiniLM-L6-v2).
> This only happens once.

---

## STEP 2: Moodle Credentials

Bilkent opens a separate Moodle site each semester, so the correct URL is critical.

### Correct URL

Current semester (2025-2026 Spring):
```
https://moodle.bilkent.edu.tr/2025-2026-spring/
```

> Warning: `moodle.bilkent.edu.tr` alone DOES NOT WORK — it's just a redirect page.
> Always use the semester-specific URL.

### Easiest Method: Username/Password

Enter your Bilkent ID and Moodle password in the `.env` file — the application will
**automatically obtain and save a token** on first run:

```env
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=your_bilkent_id
MOODLE_PASSWORD=your_moodle_password
```

When you first run `python main.py courses`:
```
Fetching Moodle token (service: moodle_mobile_app)...
Token obtained and saved to data/.moodle_token
   (You won't need to enter credentials again until token expires)
```

The token is saved to `data/.moodle_token`. No password will be asked on subsequent runs.

### "I don't know my password" Issue

If you log into Bilkent Moodle via SRS, you may not have a separate "Moodle password."
Solution steps:

**1. Try password reset:**
```
https://moodle.bilkent.edu.tr/2025-2026-spring/login/forgot_password.php
```
Enter your Bilkent email (`bilkentid@ug.bilkent.edu.tr`). If a reset email arrives,
set a new password and add it to `.env`.

**2. Log into Moodle via SRS and create a password:**
- Log into Moodle through SRS
- Top right → Profile → **Preferences** → **Change password**
- Set a new Moodle password

**3. Via the Moodle Mobile App:**
- Download the Moodle app (App Store / Play Store)
- Site URL: `https://moodle.bilkent.edu.tr/2025-2026-spring`
- Log in via SRS → the app will automatically create a token

**4. Last resort — get the token manually:**
Paste the token URL in your browser:
```
https://moodle.bilkent.edu.tr/2025-2026-spring/login/token.php?username=ID&password=PASSWORD&service=moodle_mobile_app
```
Copy the token from the returned JSON and add it to `.env`:
```env
MOODLE_TOKEN=abc123def456...
```

**5. Contact IT:**
```
moodle@bilkent.edu.tr
```
Write something like "I need an API token for an academic project."

### Troubleshooting

| Error | Solution |
|-------|----------|
| `invalidlogin` | Wrong password or direct login not enabled → reset password |
| `enablewsdescription` | Web Services disabled → contact moodle@bilkent.edu.tr |
| `servicenotavailable` | moodle_mobile_app disabled → try the Mobile App method |
| Page 404 | Wrong URL → check the semester part |
| Connection timeout | Off-campus? VPN required → https://vpn.bilkent.edu.tr |
| Token expired | Auto re-auth will be attempted; if it fails, delete `data/.moodle_token` |

---

## STEP 3: Get API Keys

Two providers are enough: GLM (chat) + OpenAI (extraction).

### GLM (Z.ai) — Main Chat Engine

1. Go to https://open.bigmodel.cn
2. Log in
3. Go to **API Keys** in the left menu
4. Click **Create API Key**
5. Copy the key

> If your GLM Coding Plan subscription is through chat.z.ai, you may need to get an API key separately.
> open.bigmodel.cn may also provide free API credits.

### OpenAI — Only for Memory Extraction ($0.05/month)

1. Go to https://platform.openai.com/api-keys
2. Log in with Google/GitHub
3. Click **+ Create new secret key**
4. Copy the key (starts with sk-...)
5. Add $5 credit at https://platform.openai.com/settings/organization/billing
   (will spend ~$0.08/month, $5 will last you ~5 years)

---

## STEP 4: Configure the .env File

```bash
cp .env.example .env
```

Then open the `.env` file with an editor:

```bash
nano .env
# or: code .env (VS Code)
# or: vim .env
```

Minimum working configuration:

```env
# ─── Moodle ──────────────────────────────────
MOODLE_URL=https://moodle.bilkent.edu.tr/2025-2026-spring
MOODLE_USERNAME=your_bilkent_id
MOODLE_PASSWORD=your_moodle_password

# ─── API Keys ────────────────────────────────
GLM_API_KEY=your_glm_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_API_KEY=sk-your_openai_key_here

# ─── Task Routing ────────────────────────────
MODEL_CHAT=glm-4.7
MODEL_EXTRACTION=gpt-4.1-nano
MODEL_TOPIC_DETECT=gpt-4.1-nano
MODEL_SUMMARY=glm-4.7
MODEL_QUESTIONS=glm-4.7
MODEL_OVERVIEW=glm-4.7
```

> On first run, the token is automatically obtained and saved to `data/.moodle_token`.
> No password will be asked again (until the token expires).

---

## STEP 5: First Run

### 5a. Test the connection

```bash
python main.py courses
```

If successful, it will list your courses:

```
Enrolled Courses

CS 453 - Computer Security (CS453-2026S)
CS 319 - Object-Oriented Software Engineering (CS319-2026S)
...
```

To see detailed content:

```bash
python main.py courses -d
```

### 5b. Synchronize content (first time takes ~5-10 min)

```bash
python main.py sync
```

This command will:
1. Scan all your courses
2. Download PDF, DOCX, PPTX files → `data/downloads/`
3. Extract text and split into chunks
4. Create embeddings and write to the FAISS index → `data/chromadb/`
5. Index forum posts
6. Update the course list in `data/profile.md`

Output:

```
Connected as 'onurcan' on 'Bilkent Moodle'
==================================================
Syncing: CS 453 - Computer Security
==================================================
[CS453] Discovered 23 document files.
Downloaded: week3_buffer_overflow.pdf (2.4MB)
Downloaded: week4_sql_injection.pdf (1.8MB)
Processed week3_buffer_overflow.pdf: 12 pages → 34 chunks
...
[CS453] Indexed 156 chunks.

Sync complete. Total new chunks indexed: 482
```

### 5c. Edit your profile (optional but recommended)

```bash
nano data/profile.md
```

Write something like:

```markdown
# Student Profile

## Identity
- Name: Onurcan
- University: Bilkent
- Department: CTIS
- Semester: Final semester (2026 Spring)

## Active Courses
<!-- auto-updated after sync -->

## Preferences
- Language: Turkish (technical terms in English in parentheses)
- Explanation style: Detailed, example-based, hands-on
- Background: Offensive security, pentest experience
- Coding language: Prefers Python

## Study Schedule
- Study in the evenings on weekdays
- Weekends for projects and bug bounty

## Long-term Goals
- Complete eWPTXv3 certification
- Specialize in AI security
- Graduate in 2026
```

This information is automatically included in the system prompt on every chat turn.

---

## STEP 6: Start Chatting

```bash
python main.py chat
```

Now you can ask:

```
You: What did we cover in Computer Security this week?
You: Summarize the SQL Injection topic
You: Generate practice questions about buffer overflow
You: Midterm is in 2 weeks, suggest a study plan
```

### Full List of Chat Commands

| Command | What It Does |
|---------|--------------|
| `/kurs <name>` | Focus on a specific course |
| `/kurslar` | List all courses |
| `/özet <course>` | Generate weekly summary |
| `/sorular <topic>` | Practice exam questions |
| `/hafıza` | Show saved memories |
| `/ilerleme` | Topic-based mastery bars |
| `/hatırla <info>` | Manually add a memory |
| `/unut <id>` | Delete a memory |
| `/profil` | View profile file |
| `/maliyet` | Estimated monthly API cost |
| `/modeller` | Active model routing table |
| `/ara <keyword>` | Search past conversations |
| `/stats` | Index + memory statistics |
| `/temizle` | Clear chat history |
| `/çıkış` | Exit |

---

## STEP 7: Web Interface (Optional)

```bash
pip install gradio
python main.py web
```

Open http://localhost:7860 in your browser.
You can also access it from your phone if you're on the same network.

For external access:

```bash
python main.py web --share
```

---

## Daily Usage

```bash
# Every day: just open chat
python main.py chat

# Once a week: sync newly added materials
python main.py sync

# Before exams: get a summary
python main.py summary -c "Computer Security"
```

---

## File Structure

```
moodle-ai-assistant/
├── .env                    ← Your API keys (in .gitignore)
├── main.py                 ← Main entry point
├── requirements.txt
├── core/
│   ├── __init__.py         ← Config management
│   ├── moodle_client.py    ← Moodle API client
│   ├── document_processor.py ← PDF/DOCX text extraction
│   ├── vector_store.py     ← ChromaDB embedding store
│   ├── llm_providers.py    ← Multi-provider (GLM/OpenAI/Claude)
│   ├── llm_engine.py       ← RAG chat engine
│   ├── memory.py           ← Hybrid memory system
│   └── sync_engine.py      ← Download + index pipeline
└── data/                   ← Auto-created
    ├── profile.md          ← Static profile (edit manually)
    ├── memory.db           ← SQLite memory DB
    ├── sync_state.json     ← Sync state
    ├── downloads/          ← Downloaded files
    └── chromadb/           ← FAISS vector index
```

---

## Troubleshooting

### "No module named 'faiss'"
```bash
pip install faiss-cpu sentence-transformers
```

### "Connection failed. Check MOODLE_URL and MOODLE_TOKEN"
- Make sure your token is correct
- Does the URL include the semester? (should be `/2025-2026-spring`)
- No trailing `/` at the end of the URL
- Off-campus? You may need **Bilkent VPN**:
  https://vpn.bilkent.edu.tr (connect with GlobalProtect client)

### "Unknown model: 'glm-4.7'"
- Check if `GLM_API_KEY` is defined in `.env`
- Is `GLM_BASE_URL` correct?

### Sync is very slow
- First sync takes a long time (downloading all files)
- Subsequent syncs only fetch new files (incremental)

### Memory extraction not working
- At least one API key must be defined
- `gpt-4.1-nano` is the cheapest option for the extraction model
- If no key is available, extraction is silently skipped (chat still works)

---

## Cost Summary

| Component | Model | Monthly Cost |
|-----------|-------|--------------|
| Chat (20 turns/day) | GLM 4.7 (subscription) | ~$0 |
| Summaries, questions | GLM 4.7 (subscription) | ~$0 |
| Memory extraction | GPT-4.1 nano | ~$0.05 |
| Topic detection | GPT-4.1 nano | ~$0.03 |
| **TOTAL** | | **~$0.08/month** |
