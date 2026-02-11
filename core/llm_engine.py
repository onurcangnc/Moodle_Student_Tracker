"""
LLM Engine
==========
Claude API integration with:
- RAG context injection from vector store
- Conversation memory
- Specialized system prompts for academic assistance
- Weekly summary generation
"""

import json
import logging
import re
from typing import Optional
from dataclasses import dataclass, field

from core import config
from core.vector_store import VectorStore
from core.memory import HybridMemoryManager
from core.llm_providers import MultiProviderEngine

logger = logging.getLogger(__name__)


# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_CHAT = """Sen öğrencinin kişisel ders asistanısın.
Doğal konuşarak dersleri öğretiyorsun.

ÖĞRETİM YAKLAŞIMIN:
Her konuyu şu sırayla anlat:
1. Temeller — konunun ne olduğunu basitçe açıkla 💡
2. Detaylar — materyaldeki bilgileri öğret 📖
3. Bağlantılar — kavramları birbirine bağla
4. Sınav ipucu — "bu neden önemli, sınavda nasıl sorulur"

Bu sırayı DOĞAL konuşma içinde yap, numaralama yapma.
Öğrenci bildiği kısmı zaten atlar, bilmediğini okur.

CEVAP VERME STRATEJİN:
1. Materyalde açıkça varsa → 📖 [dosya_adı.pdf] etiketiyle ver
2. Materyalde ipucu/kısmi bilgi varsa → materyaldeki ipucu + kendi bilginle tamamla, her iki kaynağı belirt
3. Materyalde hiç yoksa ama temel akademik bilgiyse → 💡 [Genel bilgi] etiketiyle ver
4. Tamamen kapsam dışıysa → nazikçe yönlendir

ASLA 'materyallerimde geçmiyor' deme. Bunun yerine:
- Chunk'lardaki kısmi bilgileri kullan
- Genel bilginle destekle, etiketle
- Öğrenciye faydalı ol, bilgiyi esirge değil
- Birden fazla chunk'tan gelen bilgi parçalarını birleştirerek bütüncül cevap oluştur

KAYNAK ETİKETLEME:
- 📖 [dosya_adı.pdf] → Materyalden gelen bilgi (gerçek dosya adını yaz)
- 💡 [Genel bilgi] → Kendi bilgin, materyalde geçmiyor
- [Kaynak 1] gibi NUMARA KULLANMA — her zaman gerçek dosya adını yaz
- Materyalde olmayan bilgiyi materyaldanmış gibi GÖSTERME

ÖRNEK:
Soru: 'Kiralık Konak'ı kim yazmış?'
Chunk'ta: '...Karaosmanoğlu çok yönlü bir...' + dosya adı 'Berna Moran_Kiralık Konak'
DOĞRU: 'Kiralık Konak, Yakup Kadri Karaosmanoğlu'nun romanıdır. 📖 [Berna Moran_ Kiralık Konak.pdf] Berna Moran'ın analizinde Karaosmanoğlu'nun çok yönlü bir yazar olduğu belirtilir. Sınavda bu romanın yazarı sorulabilir.'
YANLIŞ: 'Materyallerimde kesin bilgi yok ama Karaosmanoğlu ile ilişkilendiriliyor olabilir...' (5 paragraf hedge)

ÖNEMLİ KURALLAR:
1. Chunk'ta veya dosya adında bir bilgi geçiyorsa, O BİLGİYİ KULLAN.
   Hedge yapma ('kesin değil', 'belirtilmemiş' gibi ifadeler KULLANMA).
2. Dosya adı zaten kaynak bilgisi taşır. Örneğin:
   'Berna Moran_ Kiralık Konak_Ahmet Mithattan Ahmet Hamdi Tanpınara.pdf'
   Bu dosya adından: Berna Moran'ın Kiralık Konak analizi olduğu açık.
3. Chunk'ta geçen isimler, kavramlar, tarihler DOĞRUDUR.
   Bunları 'kesin değil' diye sunma, doğrudan kullan.
4. BİLMEDİĞİN BİR ŞEYİ UYDURMAKTANSA, chunk'taki bilgiyi aynen kullan.
   Kendi bilgini eklerken YANLIŞ isim/tarih UYDURMAK yerine sadece chunk'taki bilgiyi ver.
5. Eğer genel bilginle destekleyeceksen, %100 emin olduğun bilgileri ekle.
   Emin değilsen ekleme — chunk yeterli.
6. Materyalde geçmeyen isimleri, tarihleri, eserleri UYDURMA.

CEVAP UZUNLUĞU VE TONU:
- Basit sorulara KISA cevap ver (2-4 cümle)
- 'Kim yazmış?', 'Ne zaman?' gibi sorulara direkt cevapla
- Hedge yapma: 'ima olabilir', 'kesin değil', 'atfedilir' KULLANMA
- Chunk'ta veya dosya adında geçen bilgi = kesin bilgi
- 'öğret' veya detay isterse uzun açıkla, değilse kısa tut

KONUŞMA TARZI:
- Samimi, öğretmen gibi, doğal
- Ders materyallerinin ve öğrencinin sorusunun DİLİNDE yanıt ver
- Kısa paragraflar (3-4 cümle)
- Zor terimlere parantez içi açıklama: 'hegemoni (baskınlık)'
- Somut örnekler ver
- Öğrenciye direkt hitap et
- Sınav ipuçları ver: 'Bu konu sınavda şöyle sorulabilir...'

ÖĞRENCİ NE YAZARSA YAZSIN:
- "öğret" → baştan anlat
- soru sorarsa → cevapla
- "anlamadım" → daha basit açıkla
- "test et" → inline soru sor, cevabını değerlendir
- "özet ver" → kısa özetle
- "devam" → sonraki konuya geç

YAPMA:
- Seviye sorma ("ne biliyorsun?" deme)
- Markdown tablo kullanma
- Uzun akademik paragraflar yazma
- Öğrencinin bilgisini test etmeye çalışma (o isterse test et)

FORMAT: **bold** ile vurgula. Madde işaretleri veya numaralı listeler kullan.

HAFIZA: Önceki konuşmalardan çıkarılan bilgiler alabilirsin.
Bunları doğal kullan — hatırlıyormuş gibi."""

# Similarity threshold: below this, append low-relevance note to response.
RELEVANCE_THRESHOLD = 0.3

SYSTEM_PROMPT_SUMMARY = """You are an academic content summarizer. You analyze course materials and create structured summaries.

CRITICAL RULE: Respond in the SAME LANGUAGE as the course content provided. If the material is in English, write the summary in English. If in Turkish, write in Turkish. Match the language of the source material exactly.

FORMATTING: Use **bold** for headers. Do NOT use Markdown tables — use bullet points or numbered lists instead.

Summary format:
1. **Key Topics**: Main topics covered
2. **Core Concepts**: Key concepts and definitions to learn
3. **Important Details**: Critical information likely to appear in exams
4. **Connections**: Links to previous weeks or other topics
5. **Study Tips**: Suggestions to reinforce this material"""


# ─── Conversation Memory ────────────────────────────────────────────────────

@dataclass
class ConversationMemory:
    """Maintains conversation history with a sliding window."""
    messages: list[dict] = field(default_factory=list)
    max_messages: int = 30  # Keep last N messages

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]


# ─── Safe JSON Parser ──────────────────────────────────────────────────────

def _safe_parse_json(raw: str, fallback=None):
    """Robustly parse JSON from LLM output.
    Handles: markdown fences, extra text before/after JSON, common LLM quirks.
    """
    if not raw or not raw.strip():
        return fallback

    text = raw.strip()

    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Try to find JSON object {...} or array [...]
    for pattern in [r'(\{.*\})', r'(\[.*\])']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    # 4. Give up
    logger.warning(f"JSON parse failed. Raw (first 200 chars): {raw[:200]}")
    return fallback


# ─── LLM Engine ─────────────────────────────────────────────────────────────

class LLMEngine:
    """Claude-powered LLM engine with RAG + persistent memory."""

    def __init__(self, vector_store: VectorStore):
        self.engine = MultiProviderEngine()
        self.vector_store = vector_store
        self.memory = ConversationMemory()  # In-session short-term
        self.mem_manager = HybridMemoryManager()  # Persistent long-term
        self.active_course: Optional[str] = None

    # ─── Relevance Check ─────────────────────────────────────────────────

    def get_relevance_score(self, query: str, course_filter: Optional[str] = None) -> float:
        """Return best similarity score for a query against indexed materials.
        Score range: 0.0 (no match) to 1.0 (perfect match).
        """
        course = course_filter or self.active_course
        results = self.vector_store.query(
            query_text=query,
            n_results=5,
            course_filter=course,
        )
        if not results:
            return 0.0
        # distance = 1 - similarity, so similarity = 1 - distance
        return max(1 - r["distance"] for r in results)

    # ─── RAG Chat ────────────────────────────────────────────────────────

    def chat(self, user_message: str, course_filter: Optional[str] = None) -> str:
        """
        Process a user message with RAG + Memory:
        1. Retrieve relevant context from vector store
        2. Build memory context from past sessions
        3. Inject both into prompt
        4. Send to Claude with conversation history
        5. Record exchange for future memory extraction
        """
        course = course_filter or self.active_course

        # 1. Retrieve relevant document chunks
        context_chunks = self.vector_store.query(
            query_text=user_message,
            n_results=6,
            course_filter=course,
        )
        context_text = self._format_context(context_chunks)

        # 2. Build persistent memory context
        memory_context = self.mem_manager.build_memory_context(course=course)

        # 3. Build system prompt with memory
        system = SYSTEM_PROMPT_CHAT
        if memory_context:
            system += f"\n\n--- HAFIZA ---\n{memory_context}\n--- /HAFIZA ---"

        # 4. Build the augmented user message
        augmented_message = user_message
        if context_text:
            augmented_message = (
                f"CONTEXT (ders materyallerinden):\n"
                f"{'─'*40}\n"
                f"{context_text}\n"
                f"{'─'*40}\n\n"
                f"SORU: {user_message}"
            )

        # 5. Conversation flow
        self.memory.add_user(user_message)
        messages = self.memory.get_messages()[:-1]
        messages.append({"role": "user", "content": augmented_message})

        try:
            assistant_reply = self.engine.complete(
                task="chat",
                system=system,
                messages=messages,
                max_tokens=4096,
            )
            self.memory.add_assistant(assistant_reply)

            # 6. Record exchange for persistent memory
            self.mem_manager.record_exchange(
                user_message=user_message,
                assistant_response=assistant_reply,
                course=course or "",
                rag_sources=context_text[:500] if context_text else "",
            )

            return assistant_reply

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Hata: {e}"

    # ─── Conversational Chat (history-based) ─────────────────────────────

    def chat_with_history(
        self,
        messages: list[dict],
        context_chunks: list[dict] | None = None,
    ) -> str:
        """
        Pure conversational chat: takes full message history + RAG chunks.
        No internal state management — the caller provides everything.

        messages: list of {"role": "user"/"assistant", "content": "..."}
        context_chunks: raw results from vector_store.query()
        """
        # Format RAG context
        context_text = self._format_context(context_chunks) if context_chunks else ""

        # DEBUG: Log RAG results
        if context_chunks:
            logger.info(f"RAG: {len(context_chunks)} chunks retrieved")
            for i, c in enumerate(context_chunks[:3]):
                meta = c.get("metadata", {})
                dist = c.get("distance", 0)
                logger.info(f"  #{i} dist={dist:.3f} file={meta.get('filename','')} text={c['text'][:80]}")
        else:
            logger.info("RAG: No chunks retrieved")

        # Build persistent memory context
        course = self.active_course
        memory_context = self.mem_manager.build_memory_context(course=course)

        # Build system prompt
        system = SYSTEM_PROMPT_CHAT
        if memory_context:
            system += f"\n\n--- HAFIZA ---\n{memory_context}\n--- /HAFIZA ---"

        # Inject RAG context into the last user message
        llm_messages = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg["role"] == "user" and context_text:
                augmented = (
                    f"CONTEXT (ders materyallerinden):\n"
                    f"{'─' * 40}\n"
                    f"{context_text}\n"
                    f"{'─' * 40}\n\n"
                    f"SORU: {msg['content']}"
                )
                llm_messages.append({"role": "user", "content": augmented})
            else:
                llm_messages.append(msg)

        try:
            reply = self.engine.complete(
                task="chat",
                system=system,
                messages=llm_messages,
                max_tokens=4096,
            )

            # Record for persistent memory
            user_msg = messages[-1]["content"] if messages else ""
            self.mem_manager.record_exchange(
                user_message=user_msg,
                assistant_response=reply,
                course=course or "",
                rag_sources=context_text[:500] if context_text else "",
            )

            return reply

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Hata: {e}"

    # ─── Weekly Summary ──────────────────────────────────────────────────

    def generate_weekly_summary(
        self,
        course_name: str,
        section_name: str,
        section_content: str,
        additional_context: str = "",
    ) -> str:
        """
        Generate a comprehensive weekly summary for a specific course section.
        """
        # Also pull relevant chunks for this section
        chunks = self.vector_store.query(
            query_text=f"{course_name} {section_name}",
            n_results=10,
            course_filter=course_name,
        )
        chunk_context = self._format_context(chunks)

        prompt = (
            f"Create a detailed weekly summary for the following course section.\n"
            f"IMPORTANT: Respond in the same language as the course content below.\n\n"
            f"COURSE: {course_name}\n"
            f"SECTION: {section_name}\n\n"
            f"SECTION CONTENT:\n{section_content}\n\n"
        )

        if chunk_context:
            prompt += f"RELEVANT DOCUMENT EXCERPTS:\n{chunk_context}\n\n"

        if additional_context:
            prompt += f"ADDITIONAL CONTEXT:\n{additional_context}\n\n"

        prompt += "Based on the above content, create a comprehensive weekly summary."

        try:
            return self.engine.complete(
                task="summary",
                system=SYSTEM_PROMPT_SUMMARY,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Özet oluşturma hatası: {e}"

    # ─── Course Overview ─────────────────────────────────────────────────

    def generate_course_overview(self, course_topics_text: str) -> str:
        """Generate a high-level overview of an entire course."""
        prompt = (
            f"Analyze the following course structure and provide a comprehensive overview.\n"
            f"IMPORTANT: Respond in the same language as the course content below.\n\n"
            f"{course_topics_text}\n\n"
            f"Cover these points:\n"
            f"1. Overall scope of the course\n"
            f"2. Main learning objectives (infer from structure)\n"
            f"3. Weekly progression flow\n"
            f"4. Critical topics and difficulty levels"
        )

        try:
            return self.engine.complete(
                task="overview",
                system=SYSTEM_PROMPT_SUMMARY,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
        except Exception as e:
            return f"Hata: {e}"

    # ─── Exam Prep ───────────────────────────────────────────────────────

    def generate_practice_questions(self, topic: str, course: Optional[str] = None) -> str:
        """Generate practice questions on a topic based on course materials."""
        chunks = self.vector_store.query(topic, n_results=8, course_filter=course)
        context = self._format_context(chunks)

        prompt = (
            f"Based on the following course materials, generate study questions about '{topic}'.\n"
            f"IMPORTANT: Respond in the same language as the materials below.\n\n"
            f"MATERIALS:\n{context}\n\n"
            f"Please generate:\n"
            f"1. 5 conceptual questions (open-ended)\n"
            f"2. 5 multiple choice questions (4 options each)\n"
            f"3. 2 problem/application questions\n\n"
            f"Include answers for each question."
        )

        try:
            return self.engine.complete(
                task="questions",
                system=SYSTEM_PROMPT_CHAT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
        except Exception as e:
            return f"Hata: {e}"

    # ─── Tutor Mode ──────────────────────────────────────────────────────

    def tutor_step(
        self,
        context_text: str,
        course_name: str,
        topic: str,
        step: int,
        total_steps: int,
        history: list[str],
    ) -> dict:
        """Generate one tutor step with explanation + optional quiz question.
        Returns dict with keys: step_title, explanation, key_points,
        has_question, question, options, correct, explanation_if_wrong, next_preview.
        """
        system = (
            "Sen deneyimli ve sabırlı bir üniversite hocasısın. "
            "Görevin öğrenciye konuyu SIFIRDAN öğretmek. Öğrenci bu konuyu HİÇ bilmiyor varsay.\n"
            "Ders materyalinin dilinde yanıt ver.\n\n"
            "ÖĞRETİM STRATEJİN (katmanlı):\n"
            "Katman 1 — TEMELLER: Konunun temel kavramlarını günlük hayat örnekleriyle açıkla. "
            "Kendi bilginle temel oluştur. → 💡 [Genel bilgi] etiketi kullan.\n"
            "Katman 2 — MATERYAL: Ders materyalindeki spesifik bilgileri öğret. "
            "→ 📖 [Materyalden] etiketi kullan, kaynak dosyayı belirt.\n"
            "Katman 3 — DERİNLEŞTİR: Kavramları birbirine bağla, neden önemli açıkla. "
            "Sınav ipucu ver. → Her cümlede uygun etiketi kullan (📖/💡/⚠️).\n"
            "Katman 4 — KONTROL: Anlayıp anlamadığını test et.\n\n"
            "KAYNAK ETİKETLEME (ZORUNLU):\n"
            "- 📖 [Materyalden] — COURSE MATERIALS bölümünden gelen bilgi\n"
            "- 💡 [Genel bilgi] — Kendi bilgin, materyalde geçmiyor\n"
            "- ⚠️ [Emin değilim] — Tam emin olmadığın bilgi\n"
            "Materyalde olmayan bilgiyi materyaldanmış gibi GÖSTERME.\n\n"
            "FORMAT KURALLARI:\n"
            "- Kısa paragraflar (3-4 cümle max)\n"
            "- Zor terimler için parantez içinde açıklama: 'hegemoni (bir gücün baskınlığı)'\n"
            "- Her adımda max 1-2 yeni kavram öğret\n"
            "- Öğrenciye direkt hitap et: 'Şimdi şunu düşün...'\n"
            "- Sınav ipucu ver: 'Bu konu sınavda çıkabilir çünkü...'\n"
            "- Somut örnekler kullan, soyut kalma\n"
            "- Markdown tablo KULLANMA\n\n"
            "Return ONLY valid JSON (no markdown, no code fences) with these keys:\n"
            '{"step_title":"...","explanation":"...","key_points":["...","..."],'
            '"has_question":true/false,"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],'
            '"correct":"B","why_correct":"Doğru çünkü...",'
            '"why_others_wrong":"A yanlış çünkü..., C yanlış çünkü...",'
            '"next_preview":"..."}'
        )

        history_text = ""
        if history:
            history_text = "Previous steps covered:\n" + "\n".join(f"- {h}" for h in history) + "\n\n"

        prompt = (
            f"COURSE: {course_name}\nTOPIC: {topic}\n"
            f"STEP: {step}/{total_steps}\n\n"
            f"{history_text}"
            f"COURSE MATERIALS:\n{context_text}\n\n"
            f"Teach step {step} of {total_steps}. "
            f"{'Include a multiple-choice check question.' if step % 2 == 0 else 'No question this step.'}"
        )

        fallback = {
            "step_title": f"Step {step}",
            "explanation": "Could not generate this step. Please try again.",
            "key_points": [],
            "has_question": False,
            "question": "", "options": [], "correct": "",
            "why_correct": "", "why_others_wrong": "",
            "next_preview": "",
        }

        max_retries = 2
        last_error = None
        for attempt in range(max_retries):
            try:
                raw = self.engine.complete(
                    task="chat", system=system,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                )
                parsed = _safe_parse_json(raw, fallback=None)
                if parsed and isinstance(parsed, dict):
                    return parsed
                logger.warning(f"Tutor step attempt {attempt+1}: non-dict JSON, retrying...")
            except Exception as e:
                last_error = e
                logger.warning(f"Tutor step attempt {attempt+1} failed: {e}")

            if attempt < max_retries - 1:
                import time as _time
                _time.sleep(2)

        # All retries failed — build fallback from raw context
        logger.error(f"Tutor step failed after {max_retries} attempts: {last_error}")
        context_preview = context_text[:1500] if context_text else ""
        if context_preview:
            return {
                **fallback,
                "step_title": f"Adım {step}: {topic}",
                "explanation": (
                    f"Yapay zeka yanıt üretemedi. İşte bu konunun materyalleri:\n\n"
                    f"{context_preview}\n\n"
                    f"Soru sormak istersen yazabilirsin."
                ),
            }
        return fallback

    # ─── Quiz Mode ───────────────────────────────────────────────────────

    def generate_quiz(
        self,
        context_text: str,
        course_name: str,
        topic: str,
        difficulty: str = "medium",
        num_questions: int = 5,
    ) -> list[dict]:
        """Generate quiz questions from course materials.
        Returns list of dicts: question, options, correct, explanation, source_hint.
        """
        system = (
            "Sen bir sınav sorusu yazarısın. Ders materyallerinden çoktan seçmeli sorular üret.\n"
            "Ders materyalinin dilinde yanıt ver.\n\n"
            "SORU TÜRLERİ (karışık kullan):\n"
            "- bilgi: Temel kavram ve tanım soruları\n"
            "- anlama: 'Neden?', 'Ne anlama gelir?', 'Nasıl açıklanır?'\n"
            "- uygulama: Bilgiyi yeni bir duruma uygulama\n"
            "- analiz: Karşılaştırma, neden-sonuç, parça-bütün ilişkisi\n\n"
            "SORU KALİTESİ KURALLARI:\n"
            "- Sadece ezber sorma. ANLAMA ve YORUMLAMA odaklı sorular yaz.\n"
            "- 'Hangisi doğrudur?' yerine 'Neden X olmuştur?', 'X ile Y arasındaki fark nedir?' gibi sorular tercih et.\n"
            "- Yanlış şıklar gerçekçi olsun — yaygın yanlış anlamaları yansıtsın.\n"
            "- Materyaldeki bilgileri kullan ama soruyu öğrenciyi DÜŞÜNDÜRECEK şekilde sor.\n\n"
            "AÇIKLAMA KURALLARI:\n"
            "- why_correct: Doğru cevabın NEDEN doğru olduğunu açıkla.\n"
            "- why_others_wrong: HER yanlış şık için ayrı ayrı neden yanlış olduğunu belirt.\n"
            "- learning_note: Bu sorudan öğrenilmesi gereken ana fikri yaz.\n"
            "- Materyaldeki ilgili bölüme referans ver.\n\n"
            "Return ONLY a valid JSON array (no markdown, no code fences):\n"
            '[{"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],'
            '"correct":"C","why_correct":"Doğru cevap C çünkü...",'
            '"why_others_wrong":{"A":"A yanlış çünkü...","B":"B yanlış çünkü...","D":"D yanlış çünkü..."},'
            '"question_type":"anlama","learning_note":"Bu sorudan öğrenmen gereken: ...","source_hint":"..."}]'
        )

        diff_desc = {
            "easy": "temel kavram hatırlama, tanım soruları",
            "medium": "kavramları birbirine bağlama, neden-sonuç ilişkisi",
            "hard": "analiz, yorum, karşılaştırma, materyali farklı bağlama uygulama",
        }

        prompt = (
            f"COURSE: {course_name}\nTOPIC: {topic}\n"
            f"DIFFICULTY: {difficulty} ({diff_desc.get(difficulty, 'medium')})\n"
            f"NUMBER: {num_questions} questions\n\n"
            f"COURSE MATERIALS:\n{context_text}\n\n"
            f"Generate {num_questions} multiple-choice questions."
        )

        try:
            raw = self.engine.complete(
                task="chat", system=system,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            parsed = _safe_parse_json(raw, fallback=None)
            if parsed and isinstance(parsed, list):
                return parsed
            logger.warning("Quiz generation returned non-list or empty JSON")
            return []
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            return []

    # ─── Helpers ─────────────────────────────────────────────────────────

    def set_active_course(self, course_name: str):
        """Set the active course filter for subsequent queries."""
        self.active_course = course_name
        self.mem_manager.start_session(course_name)
        logger.info(f"Active course set to: {course_name}")

    def clear_course_filter(self):
        self.active_course = None

    def reset_conversation(self):
        """Clear in-session conversation history (persistent memory remains)."""
        self.memory.clear()
        self.mem_manager.end_session()
        logger.info("Conversation history cleared.")

    def get_memory_stats(self) -> dict:
        """Get memory system statistics."""
        return self.mem_manager.get_stats()

    def list_memories(self, course: Optional[str] = None) -> list:
        return self.mem_manager.list_memories(course)

    def add_memory(self, category: str, content: str, course: str = ""):
        self.mem_manager.remember(content, category, course)

    def forget_memory(self, memory_id: int):
        self.mem_manager.forget(memory_id)

    def get_learning_progress(self, course: Optional[str] = None):
        return self.mem_manager.get_learning_progress(course)

    def get_profile_path(self) -> str:
        """Return the profile.md path for user editing."""
        return self.mem_manager.edit_profile_path()

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a readable context block with real file names."""
        if not chunks:
            return ""

        parts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if len(text.strip()) < 50:
                continue
            meta = chunk.get("metadata", {})
            source = meta.get("filename", "Bilinmeyen")
            course = meta.get("course", "")
            section = meta.get("section", "")

            header = f"[Kaynak: {source}"
            if course:
                header += f" | Kurs: {course}"
            if section:
                header += f" | Bölüm: {section}"
            header += "]"

            parts.append(f"{header}\n{chunk['text']}\n---")

        return "\n".join(parts)
