"""
ingest.py — PDF ingestion pipeline for competitive intelligence RAG
Extracts text from PDFs in /pdfs, chunks them, generates embeddings,
and stores everything in pickle files for vector search.

Usage:
    python ingest.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import re
import sys
import json
import time
import pickle
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────
PDF_DIR = Path("./pdfs")
DB_DIR = Path("./vector_db")
DB_FILE = DB_DIR / "embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 800       # tokens ~= words * 1.3, so ~600 words per chunk
CHUNK_OVERLAP = 150    # overlap in characters for context continuity
BATCH_SIZE = 50        # embeddings per API call

# Known vendors — filenames should contain one of these
VENDORS = ["decagon", "sierra", "intercom", "forethought"]

# Source types — filenames should contain one of these
SOURCE_TYPES = {
    "ex-cliente": ["ex-cliente", "excliente", "customer", "cliente"],
    "ex-empleado": ["ex-empleado", "exempleado", "employee", "empleado"],
}


def detect_vendor(filename: str) -> str:
    """Detect vendor name from filename."""
    lower = filename.lower()
    for vendor in VENDORS:
        if vendor in lower:
            return vendor
    return "unknown"


def detect_source_type(filename: str) -> str:
    """Detect source type (ex-cliente or ex-empleado) from filename."""
    lower = filename.lower()
    for source_type, keywords in SOURCE_TYPES.items():
        for kw in keywords:
            if kw in lower:
                return source_type
    return "unknown"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end),
                )
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main():
    # Validate environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    if not PDF_DIR.exists():
        print(f"❌ Error: PDF directory '{PDF_DIR}' not found")
        print(f"   Create it and add your interview PDFs there.")
        sys.exit(1)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{PDF_DIR}'")
        sys.exit(1)

    print(f"📂 Found {len(pdf_files)} PDFs in {PDF_DIR}")
    print(f"🗄️  Database will be stored in {DB_DIR}")
    print()
    
    # Create database directory
    DB_DIR.mkdir(exist_ok=True)

    client = OpenAI(api_key=api_key)

    # ── Extract and chunk all PDFs ─────────────────────────────────
    all_records = []
    for i, pdf_path in enumerate(sorted(pdf_files), 1):
        filename = pdf_path.name
        vendor = detect_vendor(filename)
        source_type = detect_source_type(filename)

        print(f"  [{i}/{len(pdf_files)}] {filename}")
        print(f"           vendor={vendor}, source={source_type}")

        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"           ⚠️  No text extracted, skipping")
            continue

        chunks = chunk_text(text)
        print(f"           → {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            all_records.append({
                "text": chunk,
                "vendor": vendor,
                "source_type": source_type,
                "filename": filename,
                "chunk_index": chunk_idx,
            })

    print(f"\n📊 Total chunks to embed: {len(all_records)}")

    # ── Generate embeddings in batches ─────────────────────────────
    print(f"🧠 Generating embeddings ({EMBEDDING_MODEL})...")
    all_vectors = []
    texts_to_embed = [r["text"] for r in all_records]

    for batch_start in range(0, len(texts_to_embed), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(texts_to_embed))
        batch = texts_to_embed[batch_start:batch_end]

        print(f"   Batch {batch_start // BATCH_SIZE + 1}: "
              f"chunks {batch_start + 1}-{batch_end}")

        embeddings = get_embeddings(client, batch)
        all_vectors.extend(embeddings)

        # Rate limiting courtesy
        if batch_end < len(texts_to_embed):
            time.sleep(0.5)

    # ── Store embeddings and metadata ────────────────────────────────
    print(f"\n💾 Storing embeddings in {DB_FILE}...")

    # Prepare data structure
    database = {
        "embeddings": np.array(all_vectors, dtype=np.float32),
        "chunks": all_records,
    }
    
    # Save to pickle file
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)
    
    print(f"   ✅ Saved {len(all_records)} chunks with embeddings")

    # ── Summary ────────────────────────────────────────────────────
    print("\n✅ Ingestion complete!")
    print(f"\n📋 Summary by vendor:")
    vendor_counts = {}
    for r in all_records:
        key = f"  {r['vendor']} ({r['source_type']})"
        vendor_counts[key] = vendor_counts.get(key, 0) + 1
    for key, count in sorted(vendor_counts.items()):
        print(f"   {key}: {count} chunks")


if __name__ == "__main__":
    main()
