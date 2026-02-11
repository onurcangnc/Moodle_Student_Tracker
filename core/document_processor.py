"""
Document Processor
==================
Extracts text from various document formats (PDF, DOCX, PPTX, HTML, TXT)
and splits them into semantic chunks for embedding.
"""

import io
import logging
from pathlib import Path
from dataclasses import dataclass, field

from core import config

logger = logging.getLogger(__name__)

# OCR support (optional — graceful degrade if not installed)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class DocumentChunk:
    """A chunk of text with metadata for vector storage."""
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Unique ID based on source file + position."""
        source = self.metadata.get("source", "unknown")
        idx = self.metadata.get("chunk_index", 0)
        return f"{source}::chunk_{idx}"


class DocumentProcessor:
    """Extract text from documents and split into chunks."""

    def __init__(self):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    # ─── Main Entry Point ────────────────────────────────────────────────

    def process_file(
        self,
        file_path: Path,
        course_name: str = "",
        section_name: str = "",
        module_name: str = "",
    ) -> list[DocumentChunk]:
        """
        Extract text from a file, then chunk it.
        Returns list of DocumentChunks with rich metadata.
        """
        ext = file_path.suffix.lower()
        extractor = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".doc": self._extract_docx,   # best effort
            ".pptx": self._extract_pptx,
            ".txt": self._extract_text,
            ".md": self._extract_text,
            ".html": self._extract_html,
            ".htm": self._extract_html,
            ".rtf": self._extract_text,
        }.get(ext)

        if not extractor:
            logger.warning(f"Unsupported file format: {ext} ({file_path.name})")
            return []

        try:
            pages = extractor(file_path)
        except Exception as e:
            logger.error(f"Extraction failed [{file_path.name}]: {e}")
            return []

        if not pages:
            logger.warning(f"No text extracted from {file_path.name}")
            return []

        # Build metadata template
        base_meta = {
            "source": str(file_path),
            "filename": file_path.name,
            "course": course_name,
            "section": section_name,
            "module": module_name,
            "file_type": ext.lstrip("."),
        }

        # Chunk the pages
        chunks = self._chunk_pages(pages, base_meta)
        logger.info(f"Processed {file_path.name}: {len(pages)} pages → {len(chunks)} chunks")
        return chunks

    # ─── Extractors ──────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> list[str]:
        """Extract text from PDF using PyMuPDF (fitz) with OCR fallback for scanned pages."""
        pages = []
        ocr_pages = 0
        extended_pages = 0

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()

                # Scanned page detection: < 50 chars means likely an image
                if len(text) < 50 and OCR_AVAILABLE:
                    # _ocr_page uses two-pass: tur+eng first, extended langs if < 20 chars
                    pre_ocr_len = len(text)
                    ocr_text = self._ocr_page(page, page_num, path.name)
                    if ocr_text:
                        text = ocr_text
                        ocr_pages += 1

                if text:
                    pages.append(text)
            doc.close()

            if ocr_pages > 0:
                logger.info(f"OCR applied: {ocr_pages} page(s) in {path.name}")

            if pages:
                return pages
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyMuPDF failed, trying PyPDF2: {e}")

        # Fallback: PyPDF2 (no OCR support here)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text.strip())
        except Exception as e:
            logger.error(f"PDF extraction failed [{path.name}]: {e}")

        return pages

    def _ocr_page(self, page, page_num: int, filename: str) -> str:
        """OCR a single PDF page using Tesseract with two-pass strategy."""
        try:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            # Pass 1: Turkish + English
            try:
                text = pytesseract.image_to_string(img, lang="tur+eng")
            except Exception:
                text = pytesseract.image_to_string(img, lang="eng")

            # Pass 2: Extended languages if pass 1 was insufficient
            if len(text.strip()) < 20:
                try:
                    text = pytesseract.image_to_string(
                        img, lang="tur+eng+fra+deu+lat+ita+spa"
                    )
                    if text.strip():
                        logger.debug(f"OCR extended lang retry on page {page_num + 1} of {filename}")
                except Exception:
                    pass  # keep pass 1 result

            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed on page {page_num + 1} of {filename}: {e}")
            return ""

    def _extract_docx(self, path: Path) -> list[str]:
        """Extract text from DOCX."""
        try:
            from docx import Document
            doc = Document(str(path))
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [full_text] if full_text else []
        except Exception as e:
            logger.error(f"DOCX extraction failed [{path.name}]: {e}")
            return []

    def _extract_pptx(self, path: Path) -> list[str]:
        """Extract text from PPTX, one page per slide."""
        try:
            from pptx import Presentation
            prs = Presentation(str(path))
            slides = []
            for i, slide in enumerate(prs.slides):
                texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            t = para.text.strip()
                            if t:
                                texts.append(t)
                if texts:
                    slides.append(f"[Slide {i+1}]\n" + "\n".join(texts))
            return slides
        except Exception as e:
            logger.error(f"PPTX extraction failed [{path.name}]: {e}")
            return []

    def _extract_html(self, path: Path) -> list[str]:
        """Extract text from HTML."""
        try:
            from bs4 import BeautifulSoup
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")

            # Remove script/style tags
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return [text] if text else []
        except Exception as e:
            logger.error(f"HTML extraction failed [{path.name}]: {e}")
            return []

    def _extract_text(self, path: Path) -> list[str]:
        """Extract plain text."""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [text] if text.strip() else []
        except Exception as e:
            logger.error(f"Text extraction failed [{path.name}]: {e}")
            return []

    # ─── Chunking ────────────────────────────────────────────────────────

    def _chunk_pages(self, pages: list[str], base_meta: dict) -> list[DocumentChunk]:
        """
        Split extracted pages into overlapping chunks.
        Uses RecursiveCharacterTextSplitter if available, else manual split.
        """
        # Combine all pages with page markers
        full_text = ""
        for i, page in enumerate(pages):
            full_text += f"\n[Page {i+1}]\n{page}\n"

        chunks = self._recursive_split(full_text)

        result = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            meta = {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
            result.append(DocumentChunk(text=chunk_text.strip(), metadata=meta))

        return result

    def _recursive_split(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            return splitter.split_text(text)
        except ImportError:
            # Manual fallback
            return self._manual_split(text)

    def _manual_split(self, text: str) -> list[str]:
        """Simple chunking fallback without langchain."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    idx = text.rfind(sep, start, end)
                    if idx > start:
                        end = idx + len(sep)
                        break

            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks
