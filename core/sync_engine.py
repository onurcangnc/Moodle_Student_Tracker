"""
Sync Engine
============
Orchestrates the full pipeline:
Moodle → Download files → Process documents → Index in vector store
"""

import json
import logging
from datetime import datetime, timezone

from core import config
from core.document_processor import DocumentProcessor
from core.memory import HybridMemoryManager
from core.moodle_client import Course, MoodleClient
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SyncEngine:
    """Synchronize Moodle content to local vector store."""

    SYNC_STATE_FILE = "sync_state.json"

    def __init__(
        self,
        moodle: MoodleClient,
        processor: DocumentProcessor,
        vector_store: VectorStore,
    ):
        self.moodle = moodle
        self.processor = processor
        self.vector_store = vector_store
        self.state_file = config.data_dir / self.SYNC_STATE_FILE
        self.sync_state = self._load_state()
        self._check_semester_reset()

    # ─── Full Sync ───────────────────────────────────────────────────────

    def sync_all(self, force: bool = False):
        """
        Full sync pipeline:
        1. Fetch all enrolled courses
        2. For each course: discover + download files
        3. Process and index documents
        4. Store course structure (topics/sections) as text chunks
        """
        # Purge any leftover forum chunks (user-generated content removed for security)
        self.vector_store._delete_where(lambda m: m.get("file_type") == "forum")

        courses = self.moodle.get_courses()
        if not courses:
            logger.warning("No courses found.")
            return

        total_chunks = 0

        for course in courses:
            logger.info(f"\n{'='*50}")
            logger.info(f"Syncing: {course.fullname}")
            logger.info(f"{'='*50}")

            chunks = self.sync_course(course, force=force)
            total_chunks += chunks

        # Update profile with course list
        try:
            mem = HybridMemoryManager()
            mem.update_profile_courses([c.fullname for c in courses])
            logger.info("Profile courses updated.")
        except Exception as e:
            logger.debug(f"Profile update skipped: {e}")

        # Update sync timestamp
        self.sync_state["last_full_sync"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        logger.info(f"\n✅ Sync complete. Total new chunks indexed: {total_chunks}")

    def sync_course(self, course: Course, force: bool = False) -> int:
        """Sync a single course. Returns number of new chunks indexed."""
        chunk_count = 0

        # 1. Index course structure (sections/topics) as text
        topics_text = self.moodle.get_course_topics_text(course)
        if topics_text:
            from core.document_processor import DocumentChunk

            structure_chunk = DocumentChunk(
                text=topics_text,
                metadata={
                    "source": f"course_structure_{course.id}",
                    "filename": f"{course.shortname}_structure.md",
                    "course": course.fullname,
                    "section": "Course Structure",
                    "module": "Overview",
                    "file_type": "structure",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            )
            self.vector_store.add_chunks([structure_chunk])
            chunk_count += 1

        # 2. Download and process files
        file_results = self.moodle.download_all_course_files(course)
        logger.info(f"Downloaded {len(file_results)} files for {course.shortname}")

        for moodle_file, local_path in file_results:
            # Check if already synced (unless forced)
            file_key = str(local_path)
            if not force and file_key in self.sync_state.get("synced_files", {}):
                logger.debug(f"Skipping already synced: {local_path.name}")
                continue

            # Process and chunk the document
            chunks = self.processor.process_file(
                file_path=local_path,
                course_name=course.fullname,
                section_name=moodle_file.section_name,
                module_name=moodle_file.module_name,
            )

            if chunks:
                self.vector_store.add_chunks(chunks)
                chunk_count += len(chunks)

                # Mark as synced
                if "synced_files" not in self.sync_state:
                    self.sync_state["synced_files"] = {}
                self.sync_state["synced_files"][file_key] = {
                    "filename": moodle_file.filename,
                    "course": course.fullname,
                    "chunks": len(chunks),
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }

        # 3. Index URL modules (link + description as text chunks)
        try:
            url_modules = self.moodle.discover_url_modules(course)
            if url_modules:
                from core.document_processor import DocumentChunk

                for um in url_modules:
                    text = f"[URL: {um['name']}]\n{um['url']}"
                    if um["description"]:
                        text += f"\n{um['description']}"
                    chunk = DocumentChunk(
                        text=text,
                        metadata={
                            "source": f"url_{course.id}_{um['name'][:30]}",
                            "filename": f"url_{um['name'][:40]}",
                            "course": course.fullname,
                            "section": um["section_name"],
                            "module": um["name"],
                            "file_type": "url",
                            "chunk_index": 0,
                            "total_chunks": 1,
                        },
                    )
                    self.vector_store.add_chunks([chunk])
                    chunk_count += 1
        except Exception as e:
            logger.debug(f"URL module sync skipped: {e}")

        self._save_state()
        logger.info(f"[{course.shortname}] Indexed {chunk_count} chunks.")
        return chunk_count

    # ─── Semester Reset ─────────────────────────────────────────────────

    def _check_semester_reset(self):
        """Detect MOODLE_URL change → clear old data for new semester."""
        current_url = config.moodle_url
        saved_url = self.sync_state.get("moodle_url", "")

        if saved_url and saved_url != current_url:
            logger.info(f"🔄 Yeni dönem algılandı: {saved_url} → {current_url}")
            logger.info("Eski veriler temizleniyor...")
            # Clear vector store
            self.vector_store.reset()
            # Clear sync state
            self.sync_state = {"moodle_url": current_url}
            self._save_state()
            logger.info("✅ Dönem sıfırlama tamamlandı.")
        elif not saved_url:
            # First run — just record the URL
            self.sync_state["moodle_url"] = current_url
            self._save_state()

    # ─── State Management ────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_state(self):
        self.state_file.write_text(json.dumps(self.sync_state, indent=2, ensure_ascii=False))

    def get_sync_status(self) -> dict:
        """Return current sync status."""
        return {
            "last_full_sync": self.sync_state.get("last_full_sync", "Never"),
            "synced_files_count": len(self.sync_state.get("synced_files", {})),
            "vector_store": self.vector_store.get_stats(),
        }
