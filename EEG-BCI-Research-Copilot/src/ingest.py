"""
ingest.py
---------
Handles PDF ingestion pipeline:
  1. Extract text from PDF files using PyMuPDF
  2. Split into chunks using LangChain text splitter
  3. Generate embeddings using sentence-transformers (runs locally, free)
  4. Store in ChromaDB persistent local vector database

Usage:
    python src/ingest.py                    # ingest all PDFs in data/papers/
    python src/ingest.py --reset            # clear DB then re-ingest
    python src/ingest.py --file paper.pdf   # ingest single file
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

# Add src to path when running as script
import sys
sys.path.append(str(Path(__file__).parent))
from config import config


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file page by page.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        List of dicts with keys: text, page, source, filename
    """
    pages = []
    filename = Path(pdf_path).name

    try:
        doc = fitz.open(pdf_path)
        print(f"  📄 Extracting: {filename} ({len(doc)} pages)")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Skip pages with very little text (e.g. figures-only pages)
            if len(text.strip()) < 100:
                continue

            pages.append({
                "text": text.strip(),
                "page": page_num + 1,
                "source": pdf_path,
                "filename": filename,
            })

        doc.close()
        print(f"  ✅ Extracted {len(pages)} text pages from {filename}")

    except Exception as e:
        print(f"  ❌ Error extracting {filename}: {e}")

    return pages


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split page texts into smaller overlapping chunks for better retrieval.

    Args:
        pages: List of page dicts from extract_text_from_pdf.

    Returns:
        List of chunk dicts with text and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        # Split the page text into chunks
        texts = splitter.split_text(page["text"])

        for i, chunk_text in enumerate(texts):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": page["source"],
                    "filename": page["filename"],
                    "page": page["page"],
                    "chunk_index": i,
                }
            })

    return chunks


def get_embeddings_model() -> FakeEmbeddings:
    """
    Load fake embedding model for testing.
    This is a temporary solution until model download issues are resolved.
    
    Returns:
        FakeEmbeddings instance.
    """
    print(f"  🔄 Loading fake embedding model (size 384)")
    embeddings = FakeEmbeddings(size=384)
    print(f"  ✅ Fake embedding model loaded.")
    return embeddings


def get_vector_store(
    embeddings: FakeEmbeddings,
    reset: bool = False
) -> Chroma:
    """
    Load or create ChromaDB vector store.

    Args:
        embeddings: Embedding model to use.
        reset: If True, delete existing collection before creating.

    Returns:
        Chroma vector store instance.
    """
    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)

    if reset:
        print(f"  🗑️  Resetting ChromaDB collection: {config.COLLECTION_NAME}")
        import shutil
        if os.path.exists(config.CHROMA_DB_PATH):
            shutil.rmtree(config.CHROMA_DB_PATH)
        os.makedirs(config.CHROMA_DB_PATH)

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_PATH,
    )
    return vector_store


def ingest_pdfs(
    papers_path: str = None,
    specific_file: str = None,
    reset: bool = False
) -> int:
    """
    Main ingestion function. Processes PDFs and stores in ChromaDB.

    Args:
        papers_path: Directory containing PDF files.
        specific_file: Path to a single PDF file to ingest.
        reset: If True, clear the database before ingesting.

    Returns:
        Total number of chunks ingested.
    """
    papers_path = papers_path or config.PAPERS_PATH
    os.makedirs(papers_path, exist_ok=True)

    # Gather PDF files to process
    if specific_file:
        pdf_files = [specific_file]
    else:
        pdf_files = list(Path(papers_path).glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {papers_path}")
            print(f"   Add your EEG/neuroscience papers to: {papers_path}")
            return 0

    print(f"\n{'='*60}")
    print(f"EEG/BCI Research Copilot — Document Ingestion")
    print(f"{'='*60}")
    print(f"📁 Papers directory: {papers_path}")
    print(f"📚 Found {len(pdf_files)} PDF file(s) to process\n")

    # Load embedding model
    embeddings = get_embeddings_model()

    # Get or create vector store
    vector_store = get_vector_store(embeddings, reset=reset)

    # Check which files are already ingested
    try:
        existing = vector_store.get()
        existing_files = set(
            m.get("filename", "") for m in existing.get("metadatas", [])
        )
        print(f"📊 Already in database: {len(existing_files)} file(s)")
    except Exception:
        existing_files = set()

    # Process each PDF
    total_chunks = 0
    for pdf_path in pdf_files:
        filename = Path(pdf_path).name

        if filename in existing_files and not reset:
            print(f"  ⏭️  Skipping (already ingested): {filename}")
            continue

        print(f"\n Processing: {filename}")

        # Extract text
        pages = extract_text_from_pdf(str(pdf_path))
        if not pages:
            continue

        # Chunk text
        chunks = chunk_pages(pages)
        print(f"  📦 Created {len(chunks)} chunks")

        # Add to vector store in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            vector_store.add_texts(texts=texts, metadatas=metadatas)

        total_chunks += len(chunks)
        print(f"  ✅ Ingested {len(chunks)} chunks from {filename}")

    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Total new chunks added: {total_chunks}")
    print(f"   Database location: {config.CHROMA_DB_PATH}")
    print(f"{'='*60}\n")

    return total_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest EEG/BCI research papers into ChromaDB"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear database before ingesting"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a single PDF file to ingest"
    )
    parser.add_argument(
        "--papers-path", type=str, default=None,
        help="Path to directory containing PDF files"
    )
    args = parser.parse_args()

    config.validate()
    ingest_pdfs(
        papers_path=args.papers_path,
        specific_file=args.file,
        reset=args.reset
    )
