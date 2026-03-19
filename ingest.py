"""
ingest.py — PDF ingestion pipeline for competitive intelligence RAG
Extracts text from PDFs in /pdfs, chunks them, generates embeddings,
and stores everything in a local ChromaDB vector database.

Usage:
    python ingest.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import re
import sys
import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────
PDF_DIR = Path("./pdfs")
DB_DIR = Path("./chromadb_data")
COLLECTION_NAME = "interview_chunks"
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
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

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

    # ── Store in ChromaDB ───────────────────────────────────────────
    print(f"\n💾 Storing in ChromaDB ({DB_DIR})...")

    # Get or create collection
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"   Found existing collection '{COLLECTION_NAME}', deleting...")
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Competitive intelligence interview chunks"}
    )
    print(f"   Created collection '{COLLECTION_NAME}'")

    # Prepare data for ChromaDB (it stores embeddings separately)
    # ChromaDB expects: ids, documents, metadatas, embeddings
    DB_BATCH_SIZE = 100  # ChromaDB handles batches well
    total_batches = (len(all_records) + DB_BATCH_SIZE - 1) // DB_BATCH_SIZE
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * DB_BATCH_SIZE
        batch_end = min(batch_start + DB_BATCH_SIZE, len(all_records))
        
        # Prepare batch data
        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        batch_embeddings = []
        
        for i in range(batch_start, batch_end):
            # Create unique ID
            batch_ids.append(f"{all_records[i]['filename']}_chunk_{all_records[i]['chunk_index']}")
            batch_documents.append(all_records[i]["text"])
            batch_metadatas.append({
                "vendor": all_records[i]["vendor"],
                "source_type": all_records[i]["source_type"],
                "filename": all_records[i]["filename"],
                "chunk_index": all_records[i]["chunk_index"],
            })
            batch_embeddings.append(all_vectors[i])
        
        # Add batch to collection
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                print(f"   ✓ Added batch {batch_idx + 1}/{total_batches} ({len(batch_ids)} rows)")
        except Exception as e:
            print(f"   ❌ Error adding batch {batch_idx + 1}: {e}")
            raise
    
    # Verify collection
    try:
        count = collection.count()
        print(f"   ✅ Total rows in collection: {count}")
    except Exception as e:
        print(f"   ⚠️  Could not verify row count: {e}")

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
