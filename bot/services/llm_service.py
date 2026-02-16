"""LLM response generation for teaching-mode and guidance-mode outputs."""

from __future__ import annotations

import asyncio
import logging

from bot.services.rag_service import Chunk
from bot.state import STATE

logger = logging.getLogger(__name__)

_TEACHING_SYSTEM_PROMPT = """Sen bir universite asistanisin. Gorevin, ogrencinin sordugu soruyu asagidaki ders materyallerini kullanarak acik ve anlasilir sekilde cevaplamak.

KURALLAR:
- Cevabini SADECE verilen materyallere dayandir.
- Materyalde olmayan bilgiyi uydurma. Materyalde yoksa "Bu konu ders materyalinde yer almiyor" de.
- Ogrencinin seviyesine uygun, sade bir dil kullan.
- Gerekirse adim adim acikla.
- Gerekirse basit bir ornekle destekle.
- Cevabi tek mesajda ver, kisa ve oz tut. Paragraflar halinde yaz, madde isareti listesi yapma.
- Asla "chunk", "retrieval", "vektor", "context" gibi teknik terimler kullanma.
- Hocanin terminolojisini ve bakis acisini koru.
"""

_GUIDANCE_SYSTEM_PROMPT = """Sen bir universite asistanisin. Ogrenci bir soru sordu ama bu soru mevcut ders materyalleriyle yeterince eslesmedi.

Gorevin ogrenciyi yonlendirmek:
1. Sorunun neden materyalle eslesmedigini ACIKLAMA - sadece yardimci ol.
2. Materyalde bulunan ilgili konulari kisaca oner.
3. Daha spesifik soru sormasi icin 2-3 ornek soru ver.

KURALLAR:
- "Chunk bulunamadi" veya "yeterli eslesme yok" gibi teknik seyler soyleme.
- Dogal ve yardimci bir dil kullan.
- Ornek sorulari, materyalde gercekten karsiligi olan konulara yonlendir.
"""


def _format_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunk payload for LLM prompt consumption."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source = str(chunk.metadata.get("filename", "bilinmeyen_kaynak"))
        blocks.append(f"[Kaynak {idx}: {source}]\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def _format_topics(topics: list[str]) -> str:
    """Render available topics for guidance prompt."""
    if not topics:
        return "Bu kurs icin henuz konu basligi bulunmuyor."
    return "\n".join(f"- {topic}" for topic in topics[:12])


def _format_history(history: list[dict[str, str]]) -> str:
    """Format short conversation history into compact transcript text."""
    if not history:
        return "Yok."
    lines = [f"{item.get('role', 'user')}: {item.get('content', '').strip()}" for item in history]
    return "\n".join(lines[-5:])


async def _complete(task: str, system_prompt: str, user_prompt: str) -> str:
    """Run provider completion on a worker thread and normalize fallback errors."""
    llm = STATE.llm
    if llm is None:
        return "Sistem su an hazir degil. Lutfen birazdan tekrar deneyin."

    try:
        return await asyncio.to_thread(
            llm.engine.complete,
            task,
            system_prompt,
            [{"role": "user", "content": user_prompt}],
            1400,
        )
    except (ConnectionError, RuntimeError, TimeoutError, OSError, ValueError) as exc:
        logger.error("LLM completion failed", exc_info=True, extra={"task": task, "error": str(exc)})
        return "Su anda yanit uretemiyorum. Lutfen tekrar deneyin."


async def generate_teaching_response(
    query: str,
    chunks: list[Chunk],
    conversation_history: list[dict[str, str]],
) -> str:
    """Generate pedagojik answer grounded on retrieved lesson material."""
    system_prompt = _TEACHING_SYSTEM_PROMPT
    user_prompt = (
        f"DERS MATERYALI:\n{_format_chunks(chunks)}\n\n"
        f"ONCEKI KONUSMA:\n{_format_history(conversation_history)}\n\n"
        f"OGRENCININ SORUSU:\n{query}"
    )
    return await _complete(task="study", system_prompt=system_prompt, user_prompt=user_prompt)


async def generate_guidance_response(
    query: str,
    available_topics: list[str],
    conversation_history: list[dict[str, str]],
) -> str:
    """Generate non-technical guidance when retrieval context is insufficient."""
    system_prompt = _GUIDANCE_SYSTEM_PROMPT
    user_prompt = (
        f"MATERYALDEKI MEVCUT KONULAR:\n{_format_topics(available_topics)}\n\n"
        f"ONCEKI KONUSMA:\n{_format_history(conversation_history)}\n\n"
        f"OGRENCININ SORUSU:\n{query}"
    )
    return await _complete(task="chat", system_prompt=system_prompt, user_prompt=user_prompt)
