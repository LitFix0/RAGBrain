"""
RAGBrain — Document Loader
Section-aware chunking with PDF text cleaning.
"""

import re
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

CHUNK_SIZE = 1000
OVERLAP    = 100

SECTION_HEADERS = re.compile(
    r'^(PROJECTS|EXPERIENCE|EDUCATION|SKILLS|TECHNICAL SKILLS|CERTIFICATIONS|SUMMARY|OBJECTIVE|ACHIEVEMENTS|PUBLICATIONS)\s*$',
    re.IGNORECASE | re.MULTILINE
)


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """
    Fix PDFs where each word is on its own line.
    Joins lines that are single words or short fragments into proper sentences.
    """
    lines = text.split('\n')
    cleaned_lines = []
    buffer = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Empty line = paragraph break, flush buffer
            if buffer:
                cleaned_lines.append(' '.join(buffer))
                buffer = []
            cleaned_lines.append('')
            continue

        # Section headers — flush buffer and keep as-is
        if SECTION_HEADERS.match(stripped):
            if buffer:
                cleaned_lines.append(' '.join(buffer))
                buffer = []
            cleaned_lines.append(stripped)
            continue

        # Bullet points — flush buffer and keep as-is
        if stripped.startswith(('●', '•', '-', '*')):
            if buffer:
                cleaned_lines.append(' '.join(buffer))
                buffer = []
            cleaned_lines.append(stripped)
            continue

        # Lines with | are project headers — flush buffer and keep as-is
        if '|' in stripped:
            if buffer:
                cleaned_lines.append(' '.join(buffer))
                buffer = []
            cleaned_lines.append(stripped)
            continue

        # Short line (single word or fragment) — add to buffer to rejoin
        if len(stripped.split()) <= 3 and not stripped.endswith(('.', ':', '?', '!')):
            buffer.append(stripped)
        else:
            # Longer line — flush buffer then add this line
            if buffer:
                buffer.append(stripped)
                cleaned_lines.append(' '.join(buffer))
                buffer = []
            else:
                cleaned_lines.append(stripped)

    if buffer:
        cleaned_lines.append(' '.join(buffer))

    return '\n'.join(cleaned_lines)


def chunk_by_sections(text: str) -> List[str]:
    sections = []
    last_end = 0
    last_header = "HEADER"

    for match in SECTION_HEADERS.finditer(text):
        section_text = text[last_end:match.start()].strip()
        if section_text:
            sections.append((last_header, section_text))
        last_header = match.group().strip().upper()
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        sections.append((last_header, remaining))

    chunks = []
    for heading, content in sections:
        full = f"{heading}\n{content}"
        if len(full) <= CHUNK_SIZE:
            chunks.append(full)
        else:
            start = 0
            while start < len(content):
                piece = content[start:start + CHUNK_SIZE]
                chunks.append(f"{heading}\n{piece}".strip())
                start += CHUNK_SIZE - OVERLAP

    return [c for c in chunks if c.strip()]


def load_documents_from_dir(directory: str) -> List[Dict]:
    doc_chunks = []
    dir_path = Path(directory)
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        print(f"[DocumentLoader] No PDF files found in {directory}")
        return doc_chunks

    for pdf_file in pdf_files:
        print(f"[DocumentLoader] Loading: {pdf_file.name}")
        try:
            raw  = load_pdf(str(pdf_file))
            text = clean_text(raw)
            chunks = chunk_by_sections(text)
            print(f"[DocumentLoader]   → {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                doc_chunks.append({"source": pdf_file.name, "chunk_index": i, "text": chunk})
        except Exception as e:
            print(f"[DocumentLoader] Error: {e}")

    return doc_chunks


def load_single_pdf(file_path: str) -> List[Dict]:
    doc_chunks = []
    filename = Path(file_path).name
    try:
        raw    = load_pdf(file_path)
        text   = clean_text(raw)
        chunks = chunk_by_sections(text)
        print(f"[DocumentLoader] {filename} → {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            doc_chunks.append({"source": filename, "chunk_index": i, "text": chunk})
    except Exception as e:
        print(f"[DocumentLoader] Error: {e}")
        raise
    return doc_chunks