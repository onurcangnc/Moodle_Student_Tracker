"""
Content Tools — RAG and material access.
========================================
Tools: get_source_map, read_source, study_topic, rag_search
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from bot.services.tools import BaseTool
from bot.services.tools.helpers import resolve_course

if TYPE_CHECKING:
    from bot.state import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = ["get_content_tools"]


class GetSourceMapTool(BaseTool):
    """KATMAN 1 — Metadata aggregation + KATMAN 2 summaries."""

    @property
    def name(self) -> str:
        return "get_source_map"

    @property
    def description(self) -> str:
        return (
            "Aktif kurstaki TÜM materyallerin haritasını çıkarır. Dosya adları, chunk sayıları, "
            "hafta/konu gruplaması, dosya özetleri. 'Bu dersi çalışmak istiyorum', 'konular ne', "
            "'materyaller ne', 'neler var', 'nelere çalışabilirim' gibi isteklerde kullan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "course_filter": {
                    "type": "string",
                    "description": "Kurs adı (opsiyonel, aktif kurs kullanılır)",
                },
            },
            "required": [],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        course_name = resolve_course(args, user_id)
        if not course_name:
            return "Aktif kurs seçili değil. Önce bir kurs seç."

        store = services.vector_store
        if store is None:
            return "Materyal veritabanı hazır değil."

        try:
            files = await asyncio.to_thread(store.get_files_for_course, course_name)
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.error("Source map failed: %s", exc, exc_info=True)
            return f"Materyal haritası alınamadı: {exc}"

        if not files:
            return f"'{course_name}' kursu için yüklü materyal bulunamadı."

        from bot.services.summary_service import load_source_summary

        lines = []
        total_chunks = 0
        for f in files:
            filename = f.get("filename", "")
            chunk_count = f.get("chunk_count", 0)
            total_chunks += chunk_count
            section = f.get("section", "")

            line = f"📄 {filename} ({chunk_count} parça)"
            if section:
                line += f" — {section}"

            summary = load_source_summary(filename, course_name)
            if summary and not summary.get("fallback"):
                overview = summary.get("overview", "")
                if overview:
                    line += f"\n   Özet: {overview[:200]}"
                sections = summary.get("sections", [])
                if sections:
                    sec_names = [s.get("title", "") for s in sections[:5] if s.get("title")]
                    if sec_names:
                        line += f"\n   Bölümler: {', '.join(sec_names)}"
                difficulty = summary.get("difficulty", "")
                if difficulty:
                    line += f"\n   Seviye: {difficulty}"

            lines.append(line)

        study_order = ""
        if files:
            first_summary = load_source_summary(files[0].get("filename", ""), course_name)
            if first_summary:
                study_order = first_summary.get("suggested_study_order", "")

        header = f"📚 {course_name} — {len(files)} dosya, {total_chunks} toplam parça\n"
        result = header + "\n\n".join(lines)
        if study_order:
            result += f"\n\n💡 Önerilen çalışma sırası: {study_order}"

        return result


class ReadSourceTool(BaseTool):
    """KATMAN 2 + KATMAN 3 — Read specific source file."""

    @property
    def name(self) -> str:
        return "read_source"

    @property
    def description(self) -> str:
        return (
            "Belirli bir kaynak dosyayı OKUR. Önce hazır öğretim özetini yükler (büyük resim), "
            "sonra ilgili chunk'ları çeker (detay). Dosyayı baştan sona anlayarak gerçek öğretim "
            "yapabilirsin. 'X.pdf'i çalışayım', 'şu materyali oku', 'X dosyasını anlat' gibi "
            "isteklerde kullan. section parametresi verilirse sadece o bölümü okur."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Dosya adı (lecture_05_privacy.pdf gibi)",
                },
                "section": {
                    "type": "string",
                    "description": "Belirli bölüm/konu adı (opsiyonel — verilmezse tüm dosya özeti)",
                },
            },
            "required": ["source"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        source = args.get("source", "")
        if not source:
            return "Dosya adı belirtilmedi."

        section = args.get("section")
        course_name = resolve_course(args, user_id)

        store = services.vector_store
        if store is None:
            return "Materyal veritabanı hazır değil."

        from bot.services.summary_service import load_source_summary

        summary = load_source_summary(source, course_name or "")

        if summary and not section:
            overview = summary.get("overview", "")
            sections = summary.get("sections", [])
            cross_refs = summary.get("cross_references", [])
            study_order = summary.get("suggested_study_order", "")
            difficulty = summary.get("difficulty", "")

            parts = [f"📖 *{source}*\n"]
            if overview:
                parts.append(overview)
            if difficulty:
                parts.append(f"Seviye: {difficulty}")
            if sections:
                parts.append("\n*Bölümler:*")
                for i, s in enumerate(sections, 1):
                    title = s.get("title", f"Bölüm {i}")
                    sec_summary = s.get("summary", "")
                    concepts = s.get("key_concepts", [])
                    parts.append(f"\n{i}. *{title}*")
                    if sec_summary:
                        parts.append(f"   {sec_summary[:200]}")
                    if concepts:
                        parts.append(f"   Kavramlar: {', '.join(concepts[:6])}")
            if cross_refs:
                parts.append("\n*Bölümler arası bağlantılar:*")
                for ref in cross_refs[:5]:
                    parts.append(f"  → {ref}")
            if study_order:
                parts.append(f"\n💡 {study_order}")
            parts.append("\nHangi bölümle başlamak istersin?")

            return "\n".join(parts)

        if section:
            chunks = await asyncio.to_thread(store.get_file_chunks, source, 0)
            if not chunks:
                return f"'{source}' dosyası bulunamadı."

            sec_lower = section.lower()
            filtered = [c for c in chunks if sec_lower in c.get("text", "").lower()]
            if not filtered:
                filtered = chunks[:30]

            chunk_texts = "\n\n---\n\n".join(
                f"[Parça {c.get('chunk_index', 0) + 1}]\n{c.get('text', '')}"
                for c in filtered[:30]
                if c.get("text", "").strip()
            )

            result = ""
            if summary:
                result = f"DOSYA ÖZETİ:\n{json.dumps(summary, ensure_ascii=False)}\n\nBÖLÜM DETAYI:\n"
            result += chunk_texts
            return result

        chunks = await asyncio.to_thread(store.get_file_chunks, source, 0)
        if not chunks:
            return f"'{source}' dosyası bulunamadı. get_source_map ile doğru dosya adını kontrol et."

        if len(chunks) > 80:
            return f"Dosya çok büyük ({len(chunks)} parça). Lütfen bir bölüm belirt veya get_source_map ile bölümlere bak."

        parts = [f"📄 *{source}* — {len(chunks)} parça\n"]
        for c in chunks[:40]:
            text = c.get("text", "")
            idx = c.get("chunk_index", 0)
            if text.strip():
                parts.append(f"[Parça {idx + 1}]\n{text}")

        return "\n\n---\n\n".join(parts)


class StudyTopicTool(BaseTool):
    """Cross-source topic search with configurable depth."""

    @property
    def name(self) -> str:
        return "study_topic"

    @property
    def description(self) -> str:
        return (
            "Belirli bir konuyu TÜM kaynaklarda arar ve öğretir. read_source'dan farkı: tek dosya "
            "değil, tüm materyallerde o konuyu arar. 'Ethics nedir', 'privacy konusunu çalışayım' "
            "gibi KONU bazlı isteklerde kullan. Dosya adı belirtilmemişse bu tool'u kullan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Konu",
                },
                "depth": {
                    "type": "string",
                    "enum": ["overview", "detailed", "deep"],
                    "description": (
                        "overview: genel bakış (top-10). "
                        "detailed: detaylı (top-25, varsayılan). "
                        "deep: kapsamlı (top-50, dosya özetleri dahil)."
                    ),
                },
            },
            "required": ["topic"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        topic = args.get("topic", "")
        if not topic:
            return "Konu belirtilmedi."

        course_name = resolve_course(args, user_id)

        store = services.vector_store
        if store is None:
            return "Materyal veritabanı hazır değil."

        depth = args.get("depth", "detailed")
        top_k = {"overview": 10, "detailed": 25, "deep": 50}.get(depth, 25)

        results = await asyncio.to_thread(store.hybrid_search, topic, top_k, course_name)

        if not results and course_name:
            results = await asyncio.to_thread(store.hybrid_search, topic, top_k, None)

        if not results:
            return f"'{topic}' konusuyla ilgili materyal bulunamadı."

        parts = []
        seen_files: set[str] = set()

        if depth == "deep":
            from bot.services.summary_service import load_source_summary

        for r in results:
            meta = r.get("metadata", {})
            filename = meta.get("filename", "bilinmeyen")
            text = r.get("text", "")
            dist = r.get("distance", 0)
            if len(text.strip()) < 50:
                continue

            if depth == "deep" and filename not in seen_files:
                seen_files.add(filename)
                from bot.services.summary_service import load_source_summary
                summary = load_source_summary(filename, course_name or "")
                if summary and not summary.get("fallback"):
                    overview = summary.get("overview", "")
                    if overview:
                        parts.append(f"[📄 {filename} — Dosya Özeti: {overview[:200]}]")

            parts.append(f"[📖 {filename} | Skor: {1 - dist:.2f}]\n{text}")

        return "\n\n---\n\n".join(parts) if parts else f"'{topic}' ile ilgili yeterli materyal bulunamadı."


class RagSearchTool(BaseTool):
    """Standard RAG search for specific questions."""

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Ders materyallerinde spesifik soru/kavram arar. KISA, odaklı sorular için. "
            "Konu çalışma değil, bilgi arama. 'X nedir?', 'Y'nin tanımı ne?' gibi sorularda kullan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Soru veya kavram",
                },
                "course_name": {
                    "type": "string",
                    "description": "Kurs filtresi (opsiyonel, aktif kurs kullanılır)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, args: dict, user_id: int, services: ServiceContainer) -> str:
        query = args.get("query", "")
        if not query:
            return "Arama sorgusu belirtilmedi."

        course_name = args.get("course_name")
        if not course_name:
            course_name = resolve_course(args, user_id)

        store = services.vector_store
        if store is None:
            return "Materyal veritabanı henüz hazır değil."

        results = await asyncio.to_thread(store.hybrid_search, query, 10, course_name)

        if not results and course_name:
            results = await asyncio.to_thread(store.hybrid_search, query, 10, None)

        if not results:
            return "Bu konuyla ilgili materyal bulunamadı."

        parts = []
        for r in results:
            meta = r.get("metadata", {})
            filename = meta.get("filename", "bilinmeyen")
            course = meta.get("course", "")
            text = r.get("text", "")
            dist = r.get("distance", 0)
            if len(text.strip()) < 50:
                continue
            parts.append(f"[📖 {filename} | Kurs: {course} | Skor: {1 - dist:.2f}]\n{text}")

        return "\n\n---\n\n".join(parts) if parts else "İlgili materyal bulunamadı."


def get_content_tools() -> list[BaseTool]:
    """Factory function returning all content tools."""
    return [
        GetSourceMapTool(),
        ReadSourceTool(),
        StudyTopicTool(),
        RagSearchTool(),
    ]
