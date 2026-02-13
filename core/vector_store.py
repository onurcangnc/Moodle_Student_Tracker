"""
Vector Store
=============
FAISS-based vector store with sentence-transformers embeddings.
Stores document chunks and enables semantic retrieval for RAG.
No C++ compiler needed — faiss-cpu ships pre-built wheels.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from core import config
from core.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS vector store for course documents."""

    def __init__(self):
        self.store_dir = Path(config.chroma_dir)  # reuse same config path
        self.embedding_model_name = config.embedding_model
        self._index = None
        self._model = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._dimension: int = 0

    # ─── Persistence paths ───────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self.store_dir / "faiss.index"

    @property
    def _meta_path(self) -> Path:
        return self.store_dir / "metadata.json"

    @property
    def _legacy_pkl_path(self) -> Path:
        return self.store_dir / "metadata.pkl"

    # ─── Initialization ──────────────────────────────────────────────────

    def initialize(self):
        """Set up FAISS index and embedding model."""
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info(f"Initializing vector store at {self.store_dir}")
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Load embedding model
        self._model = SentenceTransformer(self.embedding_model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

        # Load existing index or create new
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self._ids = saved["ids"]
            self._texts = saved["texts"]
            self._metadatas = saved["metadatas"]
            logger.info(f"Vector store loaded. {len(self._ids)} chunks.")
        elif self._index_path.exists() and self._legacy_pkl_path.exists():
            # One-time migration from pickle → JSON
            logger.info("Migrating metadata from pickle to JSON...")
            self._index = faiss.read_index(str(self._index_path))
            with open(self._legacy_pkl_path, "rb") as f:
                saved = pickle.load(f)
            self._ids = saved["ids"]
            self._texts = saved["texts"]
            self._metadatas = saved["metadatas"]
            self._save()  # re-save as JSON
            self._legacy_pkl_path.unlink()  # remove old pickle file
            logger.info(f"Migration complete. {len(self._ids)} chunks. Pickle file removed.")
        else:
            self._index = faiss.IndexFlatIP(self._dimension)  # inner product (cosine after normalize)
            self._ids = []
            self._texts = []
            self._metadatas = []
            logger.info("Created new empty vector store.")

    def _save(self):
        """Persist index and metadata to disk."""
        import faiss
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ids": self._ids,
                "texts": self._texts,
                "metadatas": self._metadatas,
            }, f, ensure_ascii=False)
        # Restrict file permissions (owner-only read/write)
        try:
            os.chmod(self._meta_path, 0o600)
        except OSError:
            pass  # Windows doesn't support POSIX permissions

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # L2 normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (embeddings / norms).astype("float32")

    # ─── Indexing ────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 100):
        """Add document chunks. Skips duplicates based on chunk_id."""
        if not chunks:
            return

        existing_ids = set(self._ids)
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All chunks already indexed, skipping.")
            return

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            # Use embedding_text for encoding if available (math-normalized), original text for storage
            embed_texts = [c.embedding_text or c.text for c in batch]
            embeddings = self._encode(embed_texts)

            self._index.add(embeddings)
            for c in batch:
                self._ids.append(c.chunk_id)
                self._texts.append(c.text)
                self._metadatas.append(c.metadata)

        self._save()
        logger.info(f"Indexed {len(new_chunks)} new chunks ({len(chunks) - len(new_chunks)} duplicates skipped).")

    def delete_by_source(self, source_path: str):
        """Remove all chunks from a specific source file."""
        self._delete_where(lambda m: m.get("source") == source_path)
        logger.info(f"Deleted chunks from source: {source_path}")

    def delete_by_course(self, course_name: str):
        """Remove all chunks from a specific course."""
        self._delete_where(lambda m: m.get("course") == course_name)
        logger.info(f"Deleted all chunks for course: {course_name}")

    def _delete_where(self, predicate):
        """Rebuild index excluding items matching predicate."""
        import faiss

        keep = [i for i, m in enumerate(self._metadatas) if not predicate(m)]
        if len(keep) == len(self._ids):
            return  # nothing to delete

        if keep:
            # Reconstruct vectors for kept items
            vectors = np.vstack([self._index.reconstruct(i) for i in keep]).astype("float32")
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

        self._ids = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._save()

    # ─── Querying ────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        course_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
        filename_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """Semantic search over indexed documents."""
        if not self._ids:
            return []

        # Encode query
        query_vec = self._encode([query_text])

        # Search more than needed if filtering
        has_filter = course_filter or section_filter or filename_filter
        search_k = min(n_results * 4 if has_filter else n_results, len(self._ids))
        scores, indices = self._index.search(query_vec, search_k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            meta = self._metadatas[idx]

            # Apply filters
            if course_filter and meta.get("course") != course_filter:
                continue
            if section_filter and meta.get("section") != section_filter:
                continue
            if filename_filter and meta.get("filename") not in filename_filter:
                continue

            hits.append({
                "id": self._ids[idx],
                "text": self._texts[idx],
                "metadata": meta,
                "distance": float(1 - score),  # convert similarity to distance
            })
            if len(hits) >= n_results:
                break

        return hits

    def query_by_course_and_topic(
        self,
        topic: str,
        course_name: str,
        n_results: int = 8,
    ) -> list[dict]:
        """Convenience: search within a specific course."""
        return self.query(
            query_text=topic,
            n_results=n_results,
            course_filter=course_name,
        )

    def get_files_for_course(self, course_name: str = None) -> list[dict]:
        """Get unique files for a course with chunk counts."""
        file_counts: dict[str, int] = {}
        for meta in self._metadatas:
            if course_name and meta.get("course") != course_name:
                continue
            fname = meta.get("filename", "unknown")
            file_counts[fname] = file_counts.get(fname, 0) + 1
        return sorted(
            [{"filename": f, "chunk_count": c} for f, c in file_counts.items()],
            key=lambda x: x["chunk_count"], reverse=True,
        )

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = len(self._ids)
        courses = set()
        sources = set()
        for m in self._metadatas:
            courses.add(m.get("course", "unknown"))
            sources.add(m.get("filename", "unknown"))

        return {
            "total_chunks": count,
            "unique_courses": len(courses),
            "unique_files": len(sources),
            "courses": sorted(courses),
            "files": sorted(sources),
        }

    def reset(self):
        """Delete all data and recreate index."""
        import faiss
        self._index = faiss.IndexFlatIP(self._dimension)
        self._ids = []
        self._texts = []
        self._metadatas = []
        self._save()
        logger.info("Vector store reset.")
