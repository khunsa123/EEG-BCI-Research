"""
retriever.py
------------
Handles semantic retrieval from ChromaDB vector store.
Embeds user queries and finds the most relevant paper chunks.

Usage:
    from src.retriever import Retriever
    r = Retriever()
    results = r.search("EEG preprocessing pipeline for emotion recognition")
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

import sys
sys.path.append(str(Path(__file__).parent))
from config import config


class Retriever:
    """
    Semantic retriever for EEG/BCI research papers.
    Uses local sentence-transformer embeddings + ChromaDB.
    """

    def __init__(self):
        """Initialise the retriever with embeddings and vector store."""
        self._embeddings = None
        self._vector_store = None
        self._load()

    def _load(self) -> None:
        """Load embedding model and connect to ChromaDB."""
        print("Loading retriever...")

        # Load fake embedding model (temporary solution)
        self._embeddings = FakeEmbeddings(size=384)

        # Connect to existing ChromaDB
        self._vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=config.CHROMA_DB_PATH,
        )

        # Check how many documents are in the store
        try:
            count = self._vector_store._collection.count()
            print(f"Retriever ready. {count} chunks in database.")
        except Exception:
            print("Database appears empty. Run ingest.py first.")

    def search(
        self,
        query: str,
        k: int = None,
        filter_filename: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks matching the query.

        Args:
            query: Natural language search query.
            k: Number of results to return (default from config).
            filter_filename: If set, only return chunks from this file.

        Returns:
            List of dicts with keys: text, filename, page, score
        """
        k = k or config.TOP_K_RESULTS

        # Build filter if specific file requested
        where_filter = None
        if filter_filename:
            where_filter = {"filename": {"$eq": filter_filename}}

        try:
            # Similarity search with relevance scores
            results_with_scores = (
                self._vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=k,
                    filter=where_filter,
                )
            )
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

        # Format results
        formatted = []
        for doc, score in results_with_scores:
            formatted.append({
                "text": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", ""),
                "score": round(score, 4),
            })

        return formatted

    def search_by_paper(
        self,
        paper_name: str,
        query: str = "",
        k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all or most relevant chunks from a specific paper.

        Args:
            paper_name: Filename of the paper (e.g. "eeg_review.pdf")
            query: Optional query to rank results within the paper.
            k: Maximum chunks to return.

        Returns:
            List of chunk dicts from that paper.
        """
        if query:
            return self.search(query=query, k=k, filter_filename=paper_name)
        else:
            # Return all chunks from the paper
            try:
                results = self._vector_store.get(
                    where={"filename": {"$eq": paper_name}},
                )
                formatted = []
                for text, metadata in zip(
                    results["documents"], results["metadatas"]
                ):
                    formatted.append({
                        "text": text,
                        "filename": metadata.get("filename", "Unknown"),
                        "page": metadata.get("page", "?"),
                        "source": metadata.get("source", ""),
                        "score": 1.0,
                    })
                return formatted[:k]
            except Exception as e:
                print(f"❌ Error retrieving paper {paper_name}: {e}")
                return []

    def list_papers(self) -> List[str]:
        """
        Return list of all paper filenames currently in the database.

        Returns:
            Sorted list of unique filenames.
        """
        try:
            results = self._vector_store.get()
            filenames = list(set(
                m.get("filename", "Unknown")
                for m in results.get("metadatas", [])
            ))
            return sorted(filenames)
        except Exception as e:
            print(f"❌ Error listing papers: {e}")
            return []

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieval results into a context string for the LLM.

        Args:
            results: List of chunk dicts from search().

        Returns:
            Formatted string with source citations.
        """
        if not results:
            return "No relevant context found in the database."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] {result['filename']} (Page {result['page']}):\n"
                f"{result['text']}\n"
                f"Relevance score: {result['score']}"
            )

        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Quick test
    retriever = Retriever()

    print("\n📚 Papers in database:")
    papers = retriever.list_papers()
    for p in papers:
        print(f"  - {p}")

    if papers:
        print("\n🔍 Test search: 'EEG preprocessing ICA artifact removal'")
        results = retriever.search("EEG preprocessing ICA artifact removal")
        for r in results:
            print(f"\n  [{r['filename']} p.{r['page']}] Score: {r['score']}")
            print(f"  {r['text'][:200]}...")
