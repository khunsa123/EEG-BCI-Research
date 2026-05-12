"""
tools.py
--------
LangChain tools for the EEG/BCI Research Copilot agent.
Each tool performs a specific research task using 
retrieval + generation.

Tools available:
  1. summarize_paper    — summarise a specific paper
  2. extract_methods    — extract preprocessing/ML methods
  3. compare_datasets   — compare EEG datasets
  4. suggest_pipeline   — recommend analysis pipeline
  5. generate_citation  — generate formatted citations
  6. list_papers        — show available papers
  7. answer_question    — general Q&A (default tool)
"""

from pathlib import Path
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import sys
sys.path.append(str(Path(__file__).parent))
from config import config
from retriever import Retriever
from generator import Generator


# Shared instances — initialised once
_retriever = None
_generator = None


def get_retriever() -> Retriever:
    """Lazy initialisation of retriever."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_generator() -> Generator:
    """Lazy initialisation of generator."""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


# ── Tool Input Schemas ────────────────────────────────────────────────────────

class PaperNameInput(BaseModel):
    paper_name: str = Field(
        description="Filename of the paper (e.g. 'eeg_review.pdf'). "
                    "Use list_papers tool first if unsure."
    )


class QueryInput(BaseModel):
    query: str = Field(
        description="Natural language query or research question."
    )


class DatasetListInput(BaseModel):
    datasets: str = Field(
        description="Comma-separated list of dataset names to compare "
                    "(e.g. 'DEAP, MAHNOB-HCI, SEED')."
    )


class PipelineInput(BaseModel):
    research_question: str = Field(
        description="The specific research question or task you want "
                    "a pipeline designed for."
    )


class EmptyInput(BaseModel):
    pass


# ── Tool Implementations ──────────────────────────────────────────────────────

class SummarizePaperTool(BaseTool):
    """Summarise a specific research paper from the database."""
    name: str = "summarize_paper"
    description: str = (
        "Summarise a specific research paper. "
        "Extracts the research objective, methods, dataset, "
        "key findings, and limitations. "
        "Use this when asked to summarise or explain a specific paper. "
        "Input: paper filename (e.g. 'eeg_review.pdf')."
    )
    args_schema: Type[BaseModel] = PaperNameInput

    def _run(self, paper_name: str) -> str:
        retriever = get_retriever()
        generator = get_generator()

        chunks = retriever.search_by_paper(
            paper_name=paper_name,
            query="research objective methods dataset results findings",
            k=15,
        )

        if not chunks:
            return (
                f"❌ Paper '{paper_name}' not found in database. "
                f"Use list_papers tool to see available papers."
            )

        context = retriever.format_context(chunks)
        return generator.generate(
            query=paper_name,
            context=context,
            task_type="summarise",
        )


class ExtractMethodsTool(BaseTool):
    """Extract preprocessing pipelines and ML methods from papers."""
    name: str = "extract_methods"
    description: str = (
        "Extract EEG preprocessing pipelines, feature extraction methods, "
        "and ML architectures mentioned across research papers. "
        "Use this when asked about methods, preprocessing steps, "
        "signal processing techniques, or model architectures. "
        "Input: a query describing what methods you want to find."
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        retriever = get_retriever()
        generator = get_generator()

        # Enhance query to focus on methods
        methods_query = (
            f"preprocessing pipeline feature extraction method "
            f"architecture algorithm {query}"
        )
        chunks = retriever.search(query=methods_query, k=7)

        if not chunks:
            return "❌ No relevant methods found. Try ingesting more papers."

        context = retriever.format_context(chunks)
        return generator.generate(
            query=query,
            context=context,
            task_type="pipeline",
        )


class CompareDatasetsTool(BaseTool):
    """Compare EEG datasets mentioned in the research papers."""
    name: str = "compare_datasets"
    description: str = (
        "Compare multiple EEG or biosignal datasets mentioned in papers. "
        "Returns a comparison table with subjects, channels, tasks, "
        "sampling rate, labels, and typical use cases. "
        "Use this when asked to compare datasets. "
        "Input: comma-separated list of dataset names."
    )
    args_schema: Type[BaseModel] = DatasetListInput

    def _run(self, datasets: str) -> str:
        retriever = get_retriever()
        generator = get_generator()

        dataset_list = [d.strip() for d in datasets.split(",")]
        query = (
            f"dataset characteristics subjects channels sampling rate "
            f"labels tasks {' '.join(dataset_list)}"
        )
        chunks = retriever.search(query=query, k=8)

        if not chunks:
            return (
                f"❌ No information found about these datasets: {datasets}. "
                f"Make sure papers describing these datasets are ingested."
            )

        context = retriever.format_context(chunks)
        return generator.generate(
            query=f"Compare these EEG datasets: {datasets}",
            context=context,
            task_type="compare",
        )


class SuggestPipelineTool(BaseTool):
    """Suggest an EEG analysis pipeline for a research question."""
    name: str = "suggest_pipeline"
    description: str = (
        "Design a recommended EEG preprocessing and ML pipeline "
        "for a specific research question. Draws on methods from "
        "ingested papers to suggest validated approaches. "
        "Use this when asked to design or recommend a pipeline, "
        "or when someone asks 'how should I analyse...' questions. "
        "Input: the specific research question or task."
    )
    args_schema: Type[BaseModel] = PipelineInput

    def _run(self, research_question: str) -> str:
        retriever = get_retriever()
        generator = get_generator()

        chunks = retriever.search(
            query=research_question,
            k=config.TOP_K_RESULTS,
        )

        if not chunks:
            return (
                "❌ No relevant papers found. "
                "Please ingest papers related to your research topic first."
            )

        context = retriever.format_context(chunks)
        return generator.generate(
            query=research_question,
            context=context,
            task_type="pipeline",
        )


class GenerateCitationTool(BaseTool):
    """Generate APA and BibTeX citations for a paper."""
    name: str = "generate_citation"
    description: str = (
        "Generate formatted citations (APA and BibTeX) for a paper "
        "in the database. Extracts title, authors, year, and venue "
        "from the paper content. "
        "Input: paper filename (e.g. 'eeg_review.pdf')."
    )
    args_schema: Type[BaseModel] = PaperNameInput

    def _run(self, paper_name: str) -> str:
        retriever = get_retriever()
        generator = get_generator()

        chunks = retriever.search_by_paper(
            paper_name=paper_name,
            query="authors title journal conference year published",
            k=5,
        )

        if not chunks:
            return (
                f"❌ Paper '{paper_name}' not found. "
                f"Use list_papers to see available papers."
            )

        context = retriever.format_context(chunks)
        return generator.generate(
            query=paper_name,
            context=context,
            task_type="citation",
        )


class ListPapersTool(BaseTool):
    """List all papers currently in the database."""
    name: str = "list_papers"
    description: str = (
        "List all research papers currently available in the database. "
        "Use this first if you are unsure which papers are available, "
        "or before using summarize_paper or generate_citation tools."
    )
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        retriever = get_retriever()
        papers = retriever.list_papers()

        if not papers:
            return (
                "❌ No papers in database. "
                "Run: python src/ingest.py to add papers."
            )

        paper_list = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(papers))
        return (
            f"📚 Papers available in database ({len(papers)} total):\n"
            f"{paper_list}\n\n"
            f"You can ask me to summarise, extract methods from, "
            f"or generate citations for any of these papers."
        )


class AnswerQuestionTool(BaseTool):
    """General question answering about EEG/BCI research."""
    name: str = "answer_question"
    description: str = (
        "Answer a general research question about EEG, BCI, "
        "neuroscience, signal processing, or machine learning "
        "based on the ingested papers. "
        "This is the default tool for general questions. "
        "Input: the research question."
    )
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, query: str) -> str:
        retriever = get_retriever()
        generator = get_generator()

        chunks = retriever.search(query=query)

        if not chunks:
            return (
                "❌ No relevant information found in the database. "
                "Try ingesting more papers related to this topic."
            )

        context = retriever.format_context(chunks)
        return generator.generate(
            query=query,
            context=context,
            task_type="qa",
        )


def get_all_tools() -> list:
    """Return all available tools for the agent."""
    return [
        ListPapersTool(),
        SummarizePaperTool(),
        ExtractMethodsTool(),
        CompareDatasetsTool(),
        SuggestPipelineTool(),
        GenerateCitationTool(),
        AnswerQuestionTool(),
    ]
