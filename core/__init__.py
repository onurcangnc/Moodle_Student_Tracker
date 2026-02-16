"""Centralized configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Moodle
    moodle_url: str = os.getenv("MOODLE_URL", "")
    moodle_token: str = os.getenv("MOODLE_TOKEN", "")

    # LLM
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    # Paths
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    downloads_dir: Path = Path(os.getenv("DOWNLOADS_DIR", "./data/downloads"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./data/chromadb"))

    # Embedding & Chunking
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Summary
    summary_language: str = os.getenv("SUMMARY_LANGUAGE", "tr")

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        errors = []
        if not self.moodle_url:
            errors.append("MOODLE_URL is not set")

        # Accept either token OR username+password
        has_token = bool(self.moodle_token)
        has_credentials = bool(os.getenv("MOODLE_USERNAME")) and bool(os.getenv("MOODLE_PASSWORD"))
        has_saved_token = (self.data_dir / ".moodle_token").exists()

        if not has_token and not has_credentials and not has_saved_token:
            errors.append(
                "No Moodle auth configured. Set either:\n"
                "  - MOODLE_USERNAME + MOODLE_PASSWORD (recommended)\n"
                "  - MOODLE_TOKEN (manual)"
            )

        # At least one LLM provider needed
        has_any_llm = any(
            [
                os.getenv("GEMINI_API_KEY"),
                os.getenv("OPENAI_API_KEY"),
                os.getenv("GLM_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
            ]
        )
        if not has_any_llm:
            errors.append(
                "No LLM API key configured. Set at least one of:\n"
                "  - GEMINI_API_KEY (recommended)\n"
                "  - OPENAI_API_KEY\n"
                "  - GLM_API_KEY\n"
                "  - ANTHROPIC_API_KEY"
            )

        return errors


config = Config()
