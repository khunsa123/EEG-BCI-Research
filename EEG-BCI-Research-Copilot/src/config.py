"""
config.py
---------
Central configuration for EEG/BCI Research Copilot.
Loads environment variables and defines all settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class. All settings live here."""

    # ── Gemini API ────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE: float = 0.2               # Low temp for factual research answers
    GEMINI_MAX_TOKENS: int = 2048

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Runs locally — no API cost, no rate limits
    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L3-v2"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "eeg_papers")

    # ── Document Processing ───────────────────────────────────────────────────
    PAPERS_PATH: str = os.getenv("PAPERS_PATH", "./data/papers")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # ── Memory ───────────────────────────────────────────────────────────────
    MEMORY_WINDOW: int = 5     # Number of conversation turns to remember

    # ── System Prompts ───────────────────────────────────────────────────────
    SYSTEM_PROMPT: str = """You are an expert EEG and BCI (Brain-Computer Interface) 
research assistant with deep knowledge of neuroscience, signal processing, and 
machine learning. You help researchers understand papers, extract methods, 
compare datasets, and design experiments.

IMPORTANT RULES:
1. Base your answers ONLY on the provided context from retrieved papers.
2. Always cite your sources: mention the paper title and page number.
3. If the context does not contain enough information, say so clearly.
4. Use precise scientific language appropriate for a research context.
5. When suggesting pipelines or methods, explain the rationale.
6. Format structured outputs (tables, pipelines) clearly using markdown.
"""

    @classmethod
    def validate(cls) -> None:
        """Validate that required config values are set."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Copy .env.example to .env and add your key from "
                "https://aistudio.google.com"
            )
        print("Configuration validated successfully.")


# Singleton instance
config = Config()
