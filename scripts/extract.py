"""Text extraction from .txt, .docx, and .pdf files."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".doc", ".docx", ".pdf"}


def extract_text(file_path: Path) -> str:
    """Extract plain text from a supported file type.

    Args:
        file_path: Path to a .txt, .doc/.docx, or .pdf file.

    Returns:
        The full plain-text content of the file.

    Raises:
        ValueError: If the file type is not supported.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return _extract_txt(file_path)
    elif suffix in (".doc", ".docx"):
        return _extract_docx(file_path)
    elif suffix == ".pdf":
        return _extract_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def extract_all(input_dir: Path) -> list[tuple[Path, str]]:
    """Extract text from every supported file in a directory.

    Args:
        input_dir: Directory containing literary text files.

    Returns:
        List of (file_path, text) tuples, sorted by filename.
    """
    results: list[tuple[Path, str]] = []
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", input_dir)
        return results

    for f in files:
        logger.info("Extracting text from %s", f.name)
        try:
            text = extract_text(f)
            results.append((f, text))
        except Exception:
            logger.exception("Failed to extract text from %s", f.name)

    return results


# ---------------------------------------------------------------------------
# Private extraction helpers
# ---------------------------------------------------------------------------

def _extract_txt(path: Path) -> str:
    """Read a plain-text file, trying UTF-8 first then latin-1 fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("%s is not UTF-8, falling back to latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _extract_docx(path: Path) -> str:
    """Extract text from a .doc/.docx file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for .doc/.docx files. "
            "Install it with: pip install python-docx"
        )

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber (layout-aware)."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for .pdf files. "
            "Install it with: pip install pdfplumber"
        )

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(text)
            else:
                logger.debug("Page %d of %s yielded no text", i + 1, path.name)

    return "\n\n".join(pages)
