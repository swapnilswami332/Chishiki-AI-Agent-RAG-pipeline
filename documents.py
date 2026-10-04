import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

ALLOWED_EXTENSIONS = {".md", ".pdf", ".txt"}


class DocumentError(ValueError):
    """Raised when an uploaded document cannot be indexed."""


def clean_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DocumentError(f"Unsupported file type. Use one of: {supported}.")
    return name


def validate_upload(filename: str, content: bytes, max_bytes: int) -> str:
    safe_name = clean_filename(filename)
    if not content:
        raise DocumentError("The uploaded file is empty.")
    if len(content) > max_bytes:
        raise DocumentError(f"The uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return safe_name


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    try:
        if extension == ".pdf":
            pages = (page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
            text = "\n\n".join(pages)
        else:
            text = content.decode("utf-8")
    except (UnicodeDecodeError, OSError, ValueError, PdfReadError) as exc:
        raise DocumentError("The document could not be read.") from exc

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        raise DocumentError("No readable text was found in the document.")
    return normalized


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks
